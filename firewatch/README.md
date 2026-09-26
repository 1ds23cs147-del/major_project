# FireWatch

A fire detection console for two-channel (visible + thermal) detection, live
phone cameras, and human-confirmed dispatch. Flask on the back, plain
HTML/CSS/JS on the front, no build step.

The idea the whole thing is built around: **a flame-coloured object that is
physically cold is not a fire.** The visible channel proposes, the thermal
channel confirms or vetoes, and a person makes the final call before anyone is
sent out.

---

## See it before you install anything

`preview/` holds five near-standalone HTML files — the real templates, real
stylesheet, real JavaScript, with the network stubbed out and sample frames
inlined as data URIs. The only thing they still fetch is the Archivo webfont
from Google Fonts, so offline the type falls back to your system font and
everything else looks exactly as it will in the app.

Open `preview/live.html` in a browser and the whole interface works:
navigation, the density controls, the classifier, the review decisions, and the
alarm takeover with its siren — click *Test the alarm* on the live wall to
raise one.

Nothing in `preview/` is part of the app. Regenerate it any time with
`python make_preview.py` after you change a template or the CSS.

## Run it for real

```bash
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>.

It starts with no cameras. The live wall offers a one-click *Use this
computer's webcam* to get a feed on screen immediately; add phones from the
**Cameras** page.

If OpenCV is missing the app still boots — every page works, feeds and
detection are switched off, and a banner says so. That is deliberate, so a
broken install never leaves you with a blank screen during a demo.

## Connect your phone as a camera

1. Install **IP Webcam** (Android) or any app that serves an MJPEG stream.
2. Put the phone on the same wifi as the computer running FireWatch.
3. Open the app and tap **Start server**. It shows an address like
   `192.168.1.7:8080`.
4. In FireWatch go to **Cameras → Add a camera** and paste it. The port and the
   stream path are both optional: *Test connection* fills in `:8080` if you
   leave it off and tries the paths the common apps serve
   (`/video`, `/videofeed`, `/shot.jpg`, `/mjpegfeed?640x480`) until one
   answers, then reports the resolution and first-frame latency. If you paste
   something that already names a path, like `192.168.1.7:8080/video`, or any
   `rtsp://` URL, it is opened exactly as given and no guessing happens.
5. Repeat for as many phones as you like.

A note on browser limits: each live feed holds one HTTP connection open, and
browsers allow roughly six per host. Beyond about five simultaneous feeds the
page starts to stall. The density buttons only change how wide the grid is —
every tile keeps streaming — so to actually free a connection, pause the
cameras you are not watching from the **Cameras** page, or open extra feeds in
a second tab. This connection ceiling is also why status updates poll every two
seconds instead of using server-sent events; a persistent event stream would
eat one of those six slots.

## Pairing a thermal camera to a visible one

Live fusion needs both channels from the same scene. Add the thermal feed as its
own camera with **Channel** set to `Thermal`, then hit **Edit** on the visible
camera watching the same place and choose it under **Paired thermal camera**.
Order does not matter — you can pair after the fact, and renaming or re-pairing
does not interrupt the feed. Only changing the address restarts that capture
thread.

With a pair, every analysed frame from the visible camera is fused with the
newest thermal frame and can reach a `Fire confirmed` verdict. Without one, the
visible camera can only reach `Unverified by heat` — flagged for review, never
auto-confirmed. The label is honest about what the system actually knows.

## What the verdicts mean

| Verdict | Severity | Meaning |
|---|---|---|
| `fire` | 4 | Flame seen **and** a matching hotspot. |
| `heat_no_flame` | 3 | Hotspot, no visible flame — smouldering, or hidden behind something. |
| `needs_review` | 2 | Flame seen, heat inconclusive: warm but under the hotspot trigger, or both channels firing with a fused score still short of the alert threshold. |
| `rgb_only` | 2 | Flame-like, but no thermal image exists for this instance. |
| `flame_no_heat` | 1 | **The thermal veto.** Fire-coloured and cold: a red jacket, a sunset, a bright screen. |
| `clear` | 0 | Neither channel sees anything. |

Severity 3 and above takes over the screen with the siren. Severity 2 arrives
as a quiet toast and waits in the review queue, because an alarm that cries
wolf gets muted, and a muted alarm is worse than none. Severity 1 and 0 are not
actionable at all: they are scored and shown live, but never queued, because a
vetoed detection is not something a person needs to decide.

## How fusion actually works

Both channels return a score in `0..1`. The fused score is a weighted sum with
**thermal weighted higher on purpose** — heat is physical evidence, colour is
appearance:

```
fused = 0.45 * rgb + 0.55 * thermal
```

Then three adjustments, all in `FireDetector.fuse()` in `core/detector.py`:

- Both channels over their triggers → fused score gets a 1.08× agreement boost.
- Flame claimed but thermal below `thermal_floor` (0.35) → **veto**: the score
  is capped at 0.45, below the 0.62 alert threshold, and the verdict becomes
  `flame_no_heat`. This is the case the project exists for.
- Hotspot with no flame → `heat_no_flame`, still actionable.

That is the both-channels case. With only one channel there is nothing to fuse,
so the single score is discounted instead — ×0.8 for visible-only, ×0.85 for
thermal-only — which keeps a one-channel result visibly less confident than a
corroborated one. It is worth being able to explain that number if an examiner
asks why an `rgb_only` result at `rgb = 0.80` reports `0.64`.

Every result carries a `steps` list — the ordered tests, their values and
thresholds, and whether each passed. That is what the classifier screen renders
under *How it decided*, and it is worth showing in your viva: the system can
explain itself rather than just emitting a number.

Live detection adds temporal agreement on top. A frame counts towards an alert
only if it is actionable **and** scores at least 80 % of `alert_threshold`
(0.496 with the defaults), so a run of barely-warm `needs_review` frames never
accumulates into a full alert. An alert needs `consecutive` (4) such frames out
of the last `window` (6), including the frame in hand, and then that camera
goes quiet for `cooldown_s` (45 s). One flickering frame cannot set off the
siren.

All nine numbers are live-editable from **Classifier → Thresholds** and persist
in the database, so you can tune in front of an examiner. Eight have their own
control; the visible weight is taken as the remainder of the thermal weight, so
the two always sum to 1.

## Plugging in your trained model

There are exactly two functions to replace. The fusion logic, the live
alerting, the review queue and the entire UI keep working unchanged.

```python
RgbChannel.score(image)      -> float in 0..1   # "how flame-like is this?"
ThermalChannel.score(image)  -> float in 0..1   # "how hot is the hotspot?"
```

`image` is a BGR numpy array — exactly what `cv2.imread` and
`VideoCapture.read()` hand you. Both live in `core/detector.py`, each marked
with a `>>> REPLACE FROM HERE` block.

Until you swap them out, both channels run a colour/intensity heuristic so the
app is usable end to end. **That heuristic is scaffolding, not your project's
contribution** — it exists so the interface has something to display before
your weights are wired in. Say so if anyone asks.

For Keras models there is a working loader at the bottom of the file. Uncomment
two lines in `app.py`:

```python
from core.detector import load_keras_example
load_keras_example(detector)          # models/rgb_fire.h5, models/thermal_fire.h5
```

It assumes a single sigmoid output where 1 means fire; if yours is a 2-class
softmax, change the trailing `[0][0]` to `[0][1]` in both `rgb_score` and
`thermal_score` inside that loader. For PyTorch, ignore the helper and assign
your own functions:

```python
detector.rgb.score = my_rgb_score          # takes BGR array, returns 0..1
detector.rgb.backend = "torch"             # shown in the UI
detector.thermal.score = my_thermal_score
detector.thermal.backend = "torch"
```

If your project trains **one** model on the RGB/thermal pair rather than two
separate ones, skip the channels and override `FireDetector.classify_pair`
instead — return the same dict shape `_result()` builds and everything
downstream still works.

Uncomment the torch or tensorflow line in `requirements.txt` once you know
which you need. They are commented out so a fresh clone installs in seconds.

## The dispatch loop

Detection never dispatches anything. On an alert the app captures the visible
frame — plus the matching thermal frame if that camera has a pair — and writes
an event with status `pending`. If the verdict is severity 3 or higher the
browser takes over the screen with the siren and the captured images side by
side; severity 2 arrives as a toast and waits in **Review**. You choose
**Dispatch response team** or **False alarm**; the event moves to `dispatched`
or `dismissed` with a timestamp. Anything you skip stays in **Review**.

Alerts already waiting when you open a page do not seize the screen — they are
history, not news, and they stay in the queue. Only alerts that arrive while
you are watching trigger the takeover.

The **History** page shows the decision record, including what share of the
alerts you have already decided on turned out real — the honest measure of
whether your thresholds are set well. It reads `—` until you have decided at
least one.

*Test the alarm* on the live wall injects a synthetic alert through the same
path, so you can rehearse the demo without setting anything on fire.

## Layout

```
app.py                 routes: five pages, the video streams, the rest JSON
core/detector.py       the two channels, fusion, veto, temporal tracking
core/cameras.py        one capture thread per camera, reconnect backoff
core/store.py          sqlite: cameras, events (with their decisions), settings
static/css/app.css     design tokens and every component
static/js/core.js      polling, alarm takeover, Web Audio siren, toasts
templates/             base shell + five screens
verify.py              offline checks: templates, markup, detector, store, routes
make_preview.py        regenerates preview/
firewatch.db           created on first run
```

The palette is thermographic: cold indigo through magenta and ember to
white-hot. Colour always means temperature, on the risk meters, the verdict
slabs and the annunciator lamps, so the interface reads the same way the sensor
does.

## Checking your changes

```bash
python verify.py
```

Just over sixty checks with no server and no browser: every template renders in
ten configurations with no unrendered Jinja left behind, tags balance, ids are
unique, every `#id` that `core.js` looks up is rendered by some template, the
severity table in `core.js` still matches the detector's, every CSS custom
property resolves, the detector behaves on synthetic images (including a cold
red rectangle, which must **not** come back as fire), SQLite round-trips, and
every `/api/...` string in the JavaScript and templates — including the ones
built with `${}` — matches a route declared in `app.py`.

Worth running after you plug in your model, since the detector tests will tell
you quickly if a channel is returning something outside `0..1`.

## Things worth knowing before the demo

The database is a file — delete `firewatch.db` for a clean slate. Captured
frames pile up in `static/snapshots/`; clear it out occasionally. `app.py`
binds `0.0.0.0` so a phone on the same wifi can open the console too, which
also means anyone on that network can. Fine for a lab, not for anything real.
The siren needs one click anywhere on the page before browsers allow audio, so
click something before you start presenting.
