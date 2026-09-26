"""Offline verification: templates, detector, store, and JS syntax."""
import json, os, sys, time, sqlite3, tempfile, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FAIL = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  ok    {name}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:
        FAIL.append(name)
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc(limit=3)


# =====================================================================
print("\n[1] Templates render")
# =====================================================================
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))
env.filters["stamp"] = lambda ts: time.strftime("%d %b, %H:%M:%S", time.localtime(ts or 0))
env.filters["clock"] = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts or 0))
env.globals["url_for"] = lambda ep, **kw: "/static/" + kw.get("filename", "")

from core.detector import FusionConfig, VERDICT_META

NAV = [{"slug": s, "label": s.title(), "url": "/" + s}
       for s in ["live", "classifier", "review", "history", "cameras"]]

CAM = {"id": "a1b2c3d4", "name": "Kitchen phone", "source": "http://192.168.1.7:8080/video",
       "location": "Block B", "modality": "rgb", "pair_id": None, "enabled": True,
       "status": "online", "error": "", "fps": 14.2, "risk": 0.71, "frames": 900,
       "uptime": 320}
CAM2 = dict(CAM, id="ffff0000", name="Thermal unit", modality="thermal",
            source="0", status="offline", error="Stream dropped", risk=0.0)

EVENT = {"id": "ev001", "ts": time.time() - 90, "camera_id": "a1b2c3d4",
         "camera_name": "Kitchen phone", "location": "Block B", "source": "live",
         "verdict": "fire", "label": "Fire confirmed", "severity": 4,
         "rgb_score": 0.83, "thermal_score": 0.91, "fused_score": 0.94,
         "reason": "Both channels agree.", "steps": [], "matrix_cell": "hot_flame",
         "rgb_image": "snapshots/x_rgb.jpg", "thermal_image": None,
         "status": "pending", "decided_at": None, "note": "", "age_s": 90}
DONE = dict(EVENT, id="ev002", status="dispatched", decided_at=time.time() - 20)
DISM = dict(EVENT, id="ev003", status="dismissed", verdict="flame_no_heat",
            label="Likely false alarm", severity=1, decided_at=time.time() - 10,
            thermal_score=0.12, fused_score=0.45, rgb_image=None)

COUNTS = {"pending": 1, "dispatched": 1, "dismissed": 1, "total": 3, "top_pending": EVENT}
STATS = {
    "by_day": [{"d": "2026-09-01", "n": 5, "dispatched": 2, "dismissed": 2},
               {"d": "2026-09-02", "n": 3, "dispatched": 1, "dismissed": 1},
               {"d": "2026-09-03", "n": 1, "dispatched": 0, "dismissed": 0}],
    "by_camera": [{"cam": "Kitchen phone", "n": 7}, {"cam": "Manual upload", "n": 2}],
    "by_verdict": [{"verdict": "fire", "n": 4}],
}

BASE_CTX = dict(nav=NAV, cv_ok=True, cfg=FusionConfig().to_dict(),
                verdict_meta=VERDICT_META, counts=COUNTS)

CASES = [
    ("live.html", "live", dict(cameras=[CAM, CAM2])),
    ("live.html", "live (no cameras)", dict(cameras=[])),
    ("cameras.html", "cameras", dict(cameras=[CAM, CAM2])),
    ("cameras.html", "cameras (empty)", dict(cameras=[])),
    ("classifier.html", "classifier", {}),
    ("review.html", "review", dict(events=[EVENT, DONE, DISM])),
    ("review.html", "review (empty)", dict(events=[])),
    ("history.html", "history", dict(events=[EVENT, DONE, DISM], stats=STATS, counts=COUNTS)),
    ("history.html", "history (empty)", dict(events=[], counts=dict(COUNTS, total=0, pending=0,
                                                                   dispatched=0, dismissed=0),
                                             stats={"by_day": [], "by_camera": [], "by_verdict": []})),
]

rendered = {}
for tpl, label, extra in CASES:
    def run(tpl=tpl, extra=extra, label=label):
        # explicit kwargs win over the shared context, matching how Flask
        # merges context processors with render_template arguments
        ctx = {**BASE_CTX, **extra, "active": tpl.split(".")[0]}
        html = env.get_template(tpl).render(**ctx)
        rendered[label] = html
        assert "<html" in html and "</html>" in html, "not a full document"
        assert "{{" not in html and "{%" not in html, "unrendered jinja left in output"
        return f"{len(html):,} bytes"
    check(f"render {label}", run)

# also render the no-OpenCV notice path
check("render live (cv_ok=False shows notice)", lambda: (
    lambda h: (_ for _ in ()).throw(AssertionError("notice missing"))
    if "OpenCV is not installed" not in h else f"{len(h):,} bytes"
)(env.get_template("live.html").render(active="live", **{**BASE_CTX, "cv_ok": False},
                                       cameras=[CAM])))


# =====================================================================
print("\n[2] Markup sanity")
# =====================================================================
def balanced_tags(html):
    import re
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr", "path", "circle", "stop",
            "rect", "line", "polyline", "polygon", "use", "text"}
    # strip script/style bodies -- they contain < and > that are not markup
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S)
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.S)
    # blank out quoted attribute values so a '<' inside one is not read as a tag
    html = re.sub(r'"[^"]*"', '""', html)
    stack = []
    for m in re.finditer(r"<(/?)([a-zA-Z][\w:-]*)([^>]*?)(/?)>", html):
        closing, tag, attrs, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        if tag in void or selfclose:
            continue
        if closing:
            if not stack:
                raise AssertionError(f"</{tag}> with nothing open")
            if stack[-1] != tag:
                raise AssertionError(f"</{tag}> closes <{stack[-1]}>")
            stack.pop()
        else:
            stack.append(tag)
    if stack:
        raise AssertionError(f"never closed: {stack}")
    return "balanced"

for label, html in rendered.items():
    check(f"tags balanced: {label}", lambda h=html: balanced_tags(h))

def ids_unique(html):
    import re
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.S)
    ids = re.findall(r'\sid="([^"]+)"', html)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids: {dupes}"
    return f"{len(ids)} ids"

for label, html in rendered.items():
    check(f"ids unique: {label}", lambda h=html: ids_unique(h))

# every element the shared JS reaches for must exist in the rendered pages
def js_hooks_exist():
    """The id list is read out of core.js rather than written here by hand, so a
    lookup for an element nobody renders cannot hide behind a stale list."""
    import re
    js = open("static/js/core.js").read()
    ids = sorted(set(re.findall(r'\$\("#([\w-]+)"\)', js)))
    assert ids, "found no id lookups in core.js at all"
    everything = "".join(rendered.values())
    missing = [i for i in ids if f'id="{i}"' not in everything]
    assert not missing, f"core.js targets ids no template renders: {missing}"
    return f"{len(ids)} ids, all rendered"
check("every id core.js looks up exists", js_hooks_exist)

def js_severities_match_python():
    """core.js keeps its own severity table to colour the UI. If it drifts from
    VERDICT_META the alarm threshold silently means something different."""
    import re
    body = re.search(r"const SEV = \{(.*?)\};", open("static/js/core.js").read(), re.S)
    assert body, "SEV table not found in core.js"
    js = {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d)", body.group(1))}
    py = {k: v["severity"] for k, v in VERDICT_META.items()}
    assert js == py, f"js {js} != python {py}"
    return f"{len(py)} verdicts agree"
check("js severity table matches the detector", js_severities_match_python)

def css_vars_defined():
    css = open("static/css/app.css").read()
    import re
    used = set(re.findall(r"var\((--[\w-]+)", css))
    declared = set(re.findall(r"(?:^|[{:;])\s*(--[\w-]+)\s*:", css, re.M))
    # these are set inline from templates/JS, not in the stylesheet
    runtime = {"--v", "--t", "--cols"}
    missing = used - declared - runtime
    assert not missing, f"undeclared custom properties: {missing}"
    assert css.count("{") == css.count("}"), "unbalanced braces in CSS"
    return f"{len(declared)} tokens, {len(used)} used"
check("css custom properties resolve", css_vars_defined)


# =====================================================================
print("\n[3] Detector")
# =====================================================================
import numpy as np
import cv2
from core.detector import FireDetector, FIRE, FLAME_NO_HEAT, HEAT_NO_FLAME, CLEAR, RGB_ONLY

det = FireDetector()

def img(color, size=(240, 320)):
    a = np.zeros((size[0], size[1], 3), np.uint8)
    a[:, :] = color
    return a

def flame_photo():
    """Dark scene with a bright orange blob -- looks like fire."""
    a = img((22, 18, 16))
    cv2.circle(a, (160, 120), 55, (30, 130, 255), -1)   # BGR orange
    cv2.circle(a, (160, 120), 28, (140, 220, 255), -1)  # hot yellow core
    return a

def hot_thermal():
    a = img((40, 40, 40))
    cv2.circle(a, (160, 120), 46, (250, 250, 250), -1)
    return a

def cold_thermal():
    return img((45, 40, 60))

def red_shirt():
    """Fire-coloured but flat: the classic false positive."""
    a = img((30, 25, 25))
    cv2.rectangle(a, (90, 60), (230, 200), (40, 60, 210), -1)
    return a

def rgb_backend_handles_flame_photo():
    score = det.rgb.score(flame_photo())
    if det.rgb.backend == "heuristic":
        assert score > 0.5, f"too low: {score:.2f}"
    else:
        assert 0.0 <= score <= 1.0, f"invalid model score: {score:.2f}"
    return f"{det.rgb.backend}: {score:.2f}"
check("RGB backend handles a flame-like image", rgb_backend_handles_flame_photo)
check("blank scene scores ~0 on RGB",
      lambda: (lambda s: f"{s:.2f}" if s < 0.2 else
               (_ for _ in ()).throw(AssertionError(f"too high: {s:.2f}")))(det.rgb.score(img((30, 30, 30)))))
check("hotspot scores high on thermal",
      lambda: (lambda s: f"{s:.2f}" if s > 0.5 else
               (_ for _ in ()).throw(AssertionError(f"too low: {s:.2f}")))(det.thermal.score(hot_thermal())))
check("cold frame scores ~0 on thermal",
      lambda: (lambda s: f"{s:.2f}" if s < 0.2 else
               (_ for _ in ()).throw(AssertionError(f"too high: {s:.2f}")))(det.thermal.score(cold_thermal())))

def case(name, r, t, expect):
    def run():
        out = det.fuse(r, t)
        assert out["verdict"] == expect, f"got {out['verdict']} ({out['fused']}), wanted {expect}"
        assert 0.0 <= out["fused"] <= 1.0, f"fused out of range: {out['fused']}"
        assert out["steps"], "no decision path produced"
        return f"fused {out['fused']:.2f}, cell {out['matrix_cell']}"
    check(f"fusion: {name}", run)

case("flame + hotspot -> fire",            0.85, 0.90, FIRE)
case("flame + cold -> thermal veto",       0.92, 0.08, FLAME_NO_HEAT)
case("no flame + hotspot -> heat only",    0.10, 0.88, HEAT_NO_FLAME)
case("nothing -> clear",                   0.05, 0.05, CLEAR)
case("flame, thermal missing -> unverified", 0.80, None, RGB_ONLY)

check("thermal veto caps confidence below alert threshold", lambda: (
    lambda o: f"{o['fused']:.2f} < {det.cfg.alert_threshold}"
    if o["fused"] < det.cfg.alert_threshold else
    (_ for _ in ()).throw(AssertionError(f"veto did not cap: {o['fused']}")))(det.fuse(0.99, 0.02)))

check("thermal outweighs visible in the fused score", lambda: (
    lambda hi_t, hi_r: "thermal dominates"
    if hi_t["fused"] > hi_r["fused"] else
    (_ for _ in ()).throw(AssertionError(f"{hi_t['fused']} !> {hi_r['fused']}")))(
        det.fuse(0.30, 0.80), det.fuse(0.80, 0.30)))

def end_to_end_classification():
    result = det.classify_pair(flame_photo(), hot_thermal())
    if det.rgb.backend == "heuristic":
        assert result["verdict"] == FIRE, f"expected fire, got {result}"
    else:
        assert 0.0 <= result["fused"] <= 1.0
        assert result["thermal"] is not None
    return f"{result['verdict']} @ {result['fused']:.2f} ({det.rgb.backend})"
check("end to end: real images through classify_pair", end_to_end_classification)

check("end to end: red object with no heat is not a fire", lambda: (
    lambda o: f"{o['verdict']} @ {o['fused']:.2f}"
    if o["verdict"] != FIRE else
    (_ for _ in ()).throw(AssertionError("called a cold red object a fire")))(
        det.classify_pair(red_shirt(), cold_thermal())))

def tracker_needs_agreement():
    d = FireDetector()
    d.cfg.consecutive, d.cfg.window, d.cfg.cooldown_s = 3, 5, 0
    tr = d.make_tracker()
    fired = [bool(tr.push(flame_photo(), hot_thermal())) for _ in range(4)]
    assert fired[:2] == [False, False], f"alerted too early: {fired}"
    assert any(fired), f"never alerted: {fired}"
    return f"alerted on frame {fired.index(True) + 1} of 4"
check("live tracker waits for repeated frames", tracker_needs_agreement)

def tracker_cooldown():
    d = FireDetector()
    d.cfg.consecutive, d.cfg.window, d.cfg.cooldown_s = 1, 2, 300
    tr = d.make_tracker()
    first = tr.push(flame_photo(), hot_thermal())
    again = [tr.push(flame_photo(), hot_thermal()) for _ in range(4)]
    assert first, "did not alert at all"
    assert not any(again), "alerted again inside the cooldown"
    return "one alert per cooldown window"
check("live tracker respects cooldown", tracker_cooldown)

def tracker_alert_matches_the_frame():
    """The stored alert carries the frame in hand, so a quiet frame must never be
    the one that raises it -- not even when older frames have banked agreement."""
    d = FireDetector()
    d.cfg.consecutive, d.cfg.window, d.cfg.cooldown_s = 1, 4, 0
    tr = d.make_tracker()
    assert tr.push(flame_photo(), hot_thermal()), "did not alert on the first hit"
    d.cfg.cooldown_s = 300                      # bank a hit the cooldown swallows
    assert tr.push(flame_photo(), hot_thermal()) is None, "ignored the cooldown"
    d.cfg.cooldown_s = 0                        # cooldown over, agreement still held
    out = tr.push(img((30, 30, 30)), cold_thermal())
    assert out is None, f"a quiet frame raised an alert: {out['verdict']}"
    return "quiet frames never raise the alarm"
check("live tracker only alerts on the frame in hand", tracker_alert_matches_the_frame)

check("config update ignores junk keys and bad values", lambda: (
    lambda c: "clean" if c.rgb_trigger == 0.7 and c.w_rgb == 0.45 else
    (_ for _ in ()).throw(AssertionError(f"{c.to_dict()}")))(
        FusionConfig().update({"rgb_trigger": 0.7, "nonsense": 1, "w_rgb": "abc"})))

def agreement_cannot_exceed_window():
    """Asking for more agreeing frames than the window holds would mean the
    live tracker could never alert. The config must not allow it."""
    c = FusionConfig().update({"consecutive": 30, "window": 4})
    assert c.consecutive == 4, f"consecutive not clamped: {c.consecutive}"
    c2 = FusionConfig().update({"window": 0, "consecutive": 0, "cooldown_s": -5})
    assert c2.window == 1 and c2.consecutive == 1 and c2.cooldown_s == 0, c2.to_dict()
    # and the clamp must actually keep alerting alive
    d = FireDetector(FusionConfig().update({"consecutive": 30, "window": 4,
                                            "cooldown_s": 0}))
    tr = d.make_tracker()
    fired = [bool(tr.push(flame_photo(), hot_thermal())) for _ in range(6)]
    assert any(fired), "clamped config still never alerts"
    return f"clamped to window, alerted on frame {fired.index(True) + 1}"
check("agreement frames clamped to the window", agreement_cannot_exceed_window)


def thresholds_modal_covers_config():
    """The new classifier requires both sensor images before analysis."""
    html = open("templates/classifier.html").read()
    assert 'name="rgb"' in html and 'name="thermal"' in html
    assert 'required' in html
    return "RGB and thermal uploads are required"
check("classifier requires paired sensor images", thresholds_modal_covers_config)


# =====================================================================
print("\n[4] Store")
# =====================================================================
from core.store import Store

tmp = os.path.join(tempfile.mkdtemp(), "t.db")
st = Store(tmp)

def store_roundtrip():
    cam = st.add_camera("Phone", "http://x/video", "Lab", "rgb")
    assert [c.id for c in st.list_cameras()] == [cam.id]
    ev = st.add_event(det.fuse(0.9, 0.9), camera_id=cam.id, camera_name="Phone")
    assert ev["status"] == "pending" and ev["verdict"] == FIRE
    assert st.counts()["pending"] == 1
    assert st.counts()["top_pending"]["id"] == ev["id"]
    got = st.decide(ev["id"], "dispatched", "confirmed by hand")
    assert got["status"] == "dispatched" and got["note"] == "confirmed by hand"
    assert st.counts()["pending"] == 0 and st.counts()["dispatched"] == 1
    assert st.list_events("dispatched")[0]["id"] == ev["id"]
    assert isinstance(got["steps"], list) and got["steps"], "steps did not survive the trip"
    s = st.stats()
    assert s["by_day"] and s["by_camera"], "stats came back empty"
    st.remove_camera(cam.id)
    assert st.list_cameras() == []
    return "cameras, events, decisions, stats"
check("sqlite round trip", store_roundtrip)

def camera_edits():
    """Editing must keep pairings sane: no self-pairs, and nothing left
    pointing at a camera that is no longer a thermal channel."""
    s = Store(os.path.join(tempfile.mkdtemp(), "cam.db"))
    vis = s.add_camera("Phone", "http://a/video", "Lab", "rgb")
    th = s.add_camera("Thermal", "http://b/video", "Lab", "thermal")

    got = s.update_camera(vis.id, name="Renamed", pair_id=th.id)
    assert got.name == "Renamed" and got.pair_id == th.id, got
    assert got.source == "http://a/video", "untouched field was overwritten"

    # a partial update must not wipe the rest
    got = s.update_camera(vis.id, location="Block C")
    assert got.pair_id == th.id and got.name == "Renamed", got

    # self-pairing is impossible
    assert s.update_camera(th.id, pair_id=th.id).pair_id is None

    # a thermal camera holds no pair of its own
    assert s.update_camera(th.id, pair_id=vis.id).pair_id is None

    # demote the thermal camera -> the visible one must lose its pair
    s.update_camera(th.id, modality="rgb")
    assert next(c for c in s.list_cameras() if c.id == vis.id).pair_id is None, \
        "stale pairing survived a channel change"

    assert s.update_camera("nope", name="x") is None, "updated a missing camera"
    assert s.update_camera(vis.id, junk=1).name == "Renamed", "junk key broke the update"
    return "renames, partial updates, pairing rules"
check("camera edits keep pairings consistent", camera_edits)

check("settings persist as json", lambda: (
    lambda: (st.set_setting("fusion", {"rgb_trigger": 0.7}),
             st.get_setting("fusion")["rgb_trigger"] == 0.7
             or (_ for _ in ()).throw(AssertionError("did not persist")))
    and "ok")())

check("unknown decision is rejected",
      lambda: "rejected" if st.decide("nope", "banana") is None
      else (_ for _ in ()).throw(AssertionError("accepted a bad status")))

check("stats survive an empty database",
      lambda: (lambda s: "empty ok" if s["by_day"] == [] else
               (_ for _ in ()).throw(AssertionError(str(s))))(
          Store(os.path.join(tempfile.mkdtemp(), "e.db")).stats()))


# =====================================================================
print("\n[5] Camera helpers")
# =====================================================================
from core.cameras import suggest_urls, resolve_source, CameraSpec

check("bare host gets port and stream paths",
      lambda: "ok" if suggest_urls("192.168.1.7")[0] == "http://192.168.1.7:8080/video"
      else (_ for _ in ()).throw(AssertionError(suggest_urls("192.168.1.7")[:2])))
check("host:port is respected",
      lambda: "ok" if suggest_urls("192.168.1.7:4747")[0] == "http://192.168.1.7:4747/video"
      else (_ for _ in ()).throw(AssertionError(suggest_urls("192.168.1.7:4747")[:2])))
check("empty host yields nothing",
      lambda: "ok" if suggest_urls("") == [] else
      (_ for _ in ()).throw(AssertionError("should be empty")))
check("digit source maps to a device index",
      lambda: "ok" if CameraSpec("i", "n", "0").cv_source() == 0
      and CameraSpec("i", "n", "http://x").cv_source() == "http://x"
      else (_ for _ in ()).throw(AssertionError("bad source mapping")))

def source_resolution():
    """What people paste into the add form. A URL that already names a path must
    be opened as given; anything that is only a host gets the paths guessed."""
    cases = {
        "": ("empty", None),
        "0": ("device", "0"),
        "192.168.1.7": ("guess", "http://192.168.1.7:8080/video"),
        "192.168.1.7:8080": ("guess", "http://192.168.1.7:8080/video"),
        # what the IP Webcam app actually displays, pasted whole
        "http://192.168.1.7:8080": ("guess", "http://192.168.1.7:8080/video"),
        "http://192.168.1.7:8080/": ("guess", "http://192.168.1.7:8080/video"),
        # already names a path, so no guessing
        "http://192.168.1.7:8080/video": ("exact", "http://192.168.1.7:8080/video"),
        "192.168.1.7:8080/videofeed": ("exact", "http://192.168.1.7:8080/videofeed"),
        "rtsp://cam.local/stream1": ("exact", "rtsp://cam.local/stream1"),
    }
    for raw, (kind, first) in cases.items():
        got_kind, got = resolve_source(raw)
        assert got_kind == kind, f"{raw!r}: got {got_kind}, wanted {kind}"
        assert (got[0] if got else None) == first, f"{raw!r}: got {got[:1]}, wanted {first}"
    return f"{len(cases)} inputs classified"
check("what the user pastes resolves correctly", source_resolution)


# =====================================================================
print("\n[6] Routes declared in app.py match what the UI calls")
# =====================================================================
import re as _re

DECLARED = {(m.group(1).upper(), m.group(2)) for m in
            _re.finditer(r'@app\.(get|post|delete)\("([^"]+)"\)', open("app.py").read())}
SOURCES = ["static/js/core.js"] + [f"templates/{t}" for t in os.listdir("templates")]


def _has(path):
    """Match a concrete call against flask's <param> placeholders."""
    for _, p in DECLARED:
        pat = "^" + _re.sub(r"<[^>]+>", "[^/]+", p).replace(".", r"\.") + "$"
        if _re.match(pat, path):
            return True
    return False


def routes_match():
    called = set()
    for f in SOURCES:
        called |= set(_re.findall(r"""["'](/api/[^"'?\s]+)""", open(f).read()))
    concrete = {c.rstrip("/") for c in called if c}
    missing = sorted(p for p in concrete if not _has(p))
    assert not missing, f"UI calls routes that do not exist: {missing}"
    return f"{len(DECLARED)} routes declared, {len(concrete)} plain calls checked"
check("no UI call hits a missing route", routes_match)


def template_literal_routes():
    """URLs built with `${...}` need the placeholder filled in before they can be
    matched, but they are then checked against app.py like any other call."""
    called = set()
    for f in SOURCES:
        called |= set(_re.findall(r"`(/api/[^`]+)`", open(f).read()))
    assert called, "expected some template-literal API calls"
    concrete = sorted({_re.sub(r"\$\{[^}]*\}", "x", c).rstrip("/") for c in called})
    missing = [p for p in concrete if not _has(p)]
    assert not missing, f"dynamic call matches no route in app.py: {missing}"
    return ", ".join(concrete)
check("dynamic API calls resolve to real routes", template_literal_routes)


print("\n" + "=" * 62)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
    sys.exit(1)
print("All checks passed.")
