"""
Builds standalone preview pages: real templates, real CSS, real JS, mocked
backend. Each output file is self-contained (images inlined as data URIs) so it
can be opened by double-clicking, with no Python and no install.

    python3 make_preview.py
"""
from __future__ import annotations

import base64, json, os, re, time
import cv2
import numpy as np
from jinja2 import Environment, FileSystemLoader

from core.detector import FireDetector, FusionConfig, VERDICT_META

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "preview")
os.makedirs(OUT, exist_ok=True)
det = FireDetector()

# =========================================================================
# Stand-in camera frames. Clearly synthetic -- they exist to show the
# interface states (nominal, elevated, alarm, thermal), not to fake results.
# =========================================================================
W, H = 480, 360


def base_room(tint=(26, 22, 20), seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = tint
    # floor / wall split
    cv2.rectangle(img, (0, int(H * 0.66)), (W, H), tuple(int(c * 1.5) for c in tint), -1)
    # a lit doorway
    cv2.rectangle(img, (int(W * .06), int(H * .18)), (int(W * .3), int(H * .68)),
                  (78, 74, 68), -1)
    cv2.rectangle(img, (int(W * .08), int(H * .2)), (int(W * .28), int(H * .66)),
                  (120, 116, 108), -1)
    # crates
    for x, y, w2, h2 in [(310, 210, 96, 74), (250, 236, 62, 48), (400, 190, 62, 96)]:
        c = tuple(int(v) for v in (48 + rng.integers(0, 26), 46 + rng.integers(0, 22), 44))
        cv2.rectangle(img, (x, y), (x + w2, y + h2), c, -1)
        cv2.rectangle(img, (x, y), (x + w2, y + h2), (18, 16, 15), 1)
    # sensor noise so it reads as a camera, not a drawing
    img = cv2.add(img, rng.integers(0, 13, (H, W, 3), dtype=np.uint8))
    return img


def add_flame(img, cx=250, cy=190, scale=1.0):
    ov = img.copy()
    for r, col in [(int(52 * scale), (18, 92, 236)), (int(34 * scale), (46, 168, 252)),
                   (int(18 * scale), (168, 232, 255))]:
        cv2.circle(ov, (cx, cy), r, col, -1)
    img = cv2.addWeighted(ov, .88, img, .12, 0)
    return cv2.GaussianBlur(img, (9, 9), 0)


def add_glow(img, cx, cy, r=90):
    glow = np.zeros_like(img)
    cv2.circle(glow, (cx, cy), r, (24, 74, 150), -1)
    return cv2.addWeighted(img, 1.0, cv2.GaussianBlur(glow, (81, 81), 0), .55, 0)


def thermal_of(hot=True, cx=250, cy=190):
    """Ironbow-style false colour, the way a thermal camera renders it."""
    g = np.full((H, W), 46, np.uint8)
    cv2.rectangle(g, (0, int(H * .66)), (W, H), 62, -1)
    cv2.rectangle(g, (310, 210), (406, 284), 74, -1)
    cv2.rectangle(g, (int(W * .08), int(H * .2)), (int(W * .28), int(H * .66)), 92, -1)
    if hot:
        cv2.circle(g, (cx, cy), 54, 190, -1)
        cv2.circle(g, (cx, cy), 34, 240, -1)
        cv2.circle(g, (cx, cy), 16, 255, -1)
    g = cv2.GaussianBlur(g, (25, 25), 0)
    g = cv2.add(g, np.random.default_rng(4).integers(0, 7, (H, W), dtype=np.uint8))
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


def label(img, text, sub=""):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (W, 26), (12, 10, 9), -1)
    cv2.putText(img, text, (9, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (196, 196, 210), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (W - 9 - 9 * len(sub), 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (150, 150, 170), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (54, 50, 70), 1)
    return img


FRAMES = {
    "cam1": label(add_glow(add_flame(base_room(seed=2), 250, 196, 1.15), 250, 196),
                  "SAMPLE FRAME", "11:42:07"),
    "cam2": label(base_room((24, 24, 30), seed=5), "SAMPLE FRAME", "11:42:07"),
    "cam3": label(base_room((20, 26, 24), seed=9), "SAMPLE FRAME", "11:42:07"),
    "cam4": label(thermal_of(True), "SAMPLE THERMAL", "11:42:07"),
    "shot_rgb": label(add_glow(add_flame(base_room(seed=2), 250, 196, 1.15), 250, 196),
                      "CAPTURED AT DETECTION", "11:42:07"),
    "shot_thermal": label(thermal_of(True), "CAPTURED AT DETECTION", "11:42:07"),
    "cls_rgb": add_flame(base_room(seed=2), 250, 196, 1.2),
    "cls_thermal": thermal_of(True),
}


def data_uri(img, q=72):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


URI = {k: data_uri(v) for k, v in FRAMES.items()}

# Real detector output on the demo pair -- the preview shows genuine numbers.
DEMO_RESULT = det.classify_pair(FRAMES["cls_rgb"], FRAMES["cls_thermal"])
DEMO_RESULT["channels"] = {"rgb": {"present": True, "backend": "heuristic"},
                           "thermal": {"present": True, "backend": "heuristic"}}
COLD_RESULT = det.classify_pair(FRAMES["cls_rgb"], thermal_of(False))
COLD_RESULT["channels"] = DEMO_RESULT["channels"]

# =========================================================================
# Mock context
# =========================================================================
NAV = [{"slug": "live", "label": "Live wall", "url": "live.html"},
       {"slug": "classifier", "label": "Classifier", "url": "classifier.html"},
       {"slug": "review", "label": "Review", "url": "review.html"},
       {"slug": "history", "label": "History", "url": "history.html"},
       {"slug": "cameras", "label": "Cameras", "url": "cameras.html"}]

CAMS = [
    {"id": "cam1", "name": "Store room phone", "source": "http://192.168.1.7:8080/video",
     "location": "Block B, ground floor", "modality": "rgb", "pair_id": "cam4",
     "enabled": True, "status": "online", "error": "", "fps": 15.4, "risk": 0.86,
     "frames": 4210, "uptime": 812},
    {"id": "cam2", "name": "Corridor phone", "source": "http://192.168.1.9:8080/video",
     "location": "Block B, first floor", "modality": "rgb", "pair_id": None,
     "enabled": True, "status": "online", "error": "", "fps": 14.1, "risk": 0.07,
     "frames": 4180, "uptime": 810},
    {"id": "cam3", "name": "Built-in webcam", "source": "0", "location": "Lab desk",
     "modality": "rgb", "pair_id": None, "enabled": True, "status": "online",
     "error": "", "fps": 29.6, "risk": 0.21, "frames": 8800, "uptime": 806},
    {"id": "cam4", "name": "Thermal unit", "source": "http://192.168.1.11:8080/video",
     "location": "Block B, ground floor", "modality": "thermal", "pair_id": None,
     "enabled": True, "status": "online", "error": "", "fps": 8.9, "risk": 0.0,
     "frames": 2400, "uptime": 800},
    {"id": "cam5", "name": "Yard phone", "source": "http://192.168.1.14:8080/video",
     "location": "Rear yard", "modality": "rgb", "pair_id": None, "enabled": True,
     "status": "offline", "error": "Stream dropped, retrying", "fps": 0.0,
     "risk": 0.0, "frames": 300, "uptime": 0},
]

CAMS = []
NOW = time.time()


def ev(i, **kw):
    base = {"id": f"ev{i}", "ts": NOW - 60 * i, "camera_id": "cam1",
            "camera_name": "Store room phone", "location": "Block B, ground floor",
            "source": "live", "verdict": "fire", "label": "Fire confirmed",
            "severity": 4, "rgb_score": 0.84, "thermal_score": 0.92,
            "fused_score": 0.95, "reason": VERDICT_META["fire"]["label"],
            "steps": [], "matrix_cell": "hot_flame",
            "rgb_image": "snapshots/shot_rgb.jpg",
            "thermal_image": "snapshots/shot_thermal.jpg",
            "status": "pending", "decided_at": None, "note": "", "age_s": 60 * i}
    base.update(kw)
    base["reason"] = {
        "fire": "Both channels agree: visible flame with a matching heat signature.",
        "flame_no_heat": "Flame-coloured, but cold. Almost certainly a fire-coloured "
                         "object rather than a fire.",
        "heat_no_flame": "A hotspot with no visible flame. Could be smouldering, or a "
                         "source hidden behind an obstruction.",
        "rgb_only": "Flame-like appearance, but with no thermal image this cannot be "
                    "confirmed. Sent for human review.",
    }.get(base["verdict"], base["reason"])
    return base


# The preview has no connected cameras; live tiles appear only in the real app
# after a source is added and produces a working frame.
CAMS = []

PENDING = [
   ev(1),
   ev(2, verdict="heat_no_flame", label="Heat, no flame", severity=3,
     rgb_score=0.19, thermal_score=0.88, fused_score=0.68,
     matrix_cell="hot_noflame", camera_name="Corridor phone",
     location="Block B, first floor", rgb_image="snapshots/shot_rgb.jpg"),
   ev(3, verdict="rgb_only", label="Unverified by heat", severity=2,
     rgb_score=0.71, thermal_score=None, fused_score=0.57,
     matrix_cell="unknown_flame", camera_name="Yard phone", location="Rear yard",
     thermal_image=None),
]
ALL_EVENTS = PENDING + [
   ev(9, status="dispatched", decided_at=NOW - 480),
   ev(11, verdict="flame_no_heat", label="Likely false alarm", severity=1,
     rgb_score=0.79, thermal_score=0.09, fused_score=0.45,
     matrix_cell="cold_flame", status="dismissed", decided_at=NOW - 900,
     camera_name="Corridor phone", note="Someone's red jacket on a chair."),
   ev(14, status="dispatched", decided_at=NOW - 1500, camera_name="Manual upload",
     source="upload", location=""),
   ev(19, verdict="flame_no_heat", label="Likely false alarm", severity=1,
     rgb_score=0.83, thermal_score=0.14, fused_score=0.45,
     matrix_cell="cold_flame", status="dismissed", decided_at=NOW - 2400,
     camera_name="Yard phone", location="Rear yard"),
]

COUNTS = {"pending": 3, "dispatched": 2, "dismissed": 2, "total": 7,
       "top_pending": PENDING[0]}

day = lambda n: time.strftime("%Y-%m-%d", time.localtime(NOW - n * 86400))
STATS = {"by_day": [{"d": day(6), "n": 4, "dispatched": 1, "dismissed": 3},
                    {"d": day(5), "n": 2, "dispatched": 1, "dismissed": 1},
                    {"d": day(4), "n": 6, "dispatched": 2, "dismissed": 4},
                    {"d": day(3), "n": 1, "dispatched": 0, "dismissed": 1},
                    {"d": day(2), "n": 5, "dispatched": 3, "dismissed": 2},
                    {"d": day(1), "n": 3, "dispatched": 1, "dismissed": 2},
                    {"d": day(0), "n": 7, "dispatched": 2, "dismissed": 2}],
         "by_camera": [{"cam": "Store room phone", "n": 11},
                       {"cam": "Corridor phone", "n": 7},
                       {"cam": "Yard phone", "n": 5},
                       {"cam": "Manual upload", "n": 3},
                       {"cam": "Built-in webcam", "n": 2}],
         "by_verdict": [{"verdict": "fire", "n": 9}]}

# =========================================================================
# Render
# =========================================================================
SNAP_MAP = {"snapshots/shot_rgb.jpg": URI["shot_rgb"],
            "snapshots/shot_thermal.jpg": URI["shot_thermal"]}


def url_for(endpoint, **kw):
    f = kw.get("filename", "")
    if f in SNAP_MAP:
        return SNAP_MAP[f]
    return "__STATIC__/" + f


env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")))
env.filters["stamp"] = lambda ts: time.strftime("%d %b, %H:%M:%S", time.localtime(ts or 0))
env.filters["clock"] = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts or 0))
env.globals["url_for"] = url_for

CSS = open(os.path.join(HERE, "static/css/app.css")).read()
CORE_JS = open(os.path.join(HERE, "static/js/core.js")).read()

MOCK_JS = """
(() => {
  const CAMS = __CAMS__;
  let counts = __COUNTS__;
  const PENDING = __PENDING__;
  const RESULTS = { hot: __DEMO__, cold: __COLD__ };
  let drift = 0, injected = null;

  const reply = (data) => Promise.resolve({
    ok: true, status: 200, json: async () => data,
  });

  window.fetch = (url, opts = {}) => {
    const method = (opts.method || "GET").toUpperCase();
    url = String(url);

    if (url.startsWith("/api/state")) {
      drift += 1;
      const cams = CAMS.map((c) => {
        if (c.status !== "online" || c.risk === 0) return c;
        // gentle wander so the meters and lamps visibly live
        const wobble = Math.sin(drift / 6 + c.id.charCodeAt(3)) * 0.05;
        return { ...c, risk: Math.max(0, Math.min(1, c.risk + wobble)) };
      });
      return reply({
        cameras: cams, counts,
        online: cams.filter((c) => c.status === "online").length,
        total: cams.length,
        peak_risk: Math.max(...cams.map((c) => c.risk)),
        armed: true, cv_ok: true, server_time: Date.now() / 1000,
      });
    }

    if (url === "/api/test-alert" && method === "POST") {
      injected = { ...PENDING[0], id: "preview-" + Date.now(), ts: Date.now() / 1000 };
      counts = { ...counts, pending: counts.pending + 1, total: counts.total + 1,
                 top_pending: injected };
      return reply({ event: injected, counts });
    }

    const decide = url.match(/^\\/api\\/events\\/([^/]+)\\/(dispatch|dismiss|reopen)$/);
    if (decide && method === "POST") {
      const key = decide[2] === "dispatch" ? "dispatched" : "dismissed";
      counts = { ...counts, pending: Math.max(0, counts.pending - 1),
                 [key]: (counts[key] || 0) + 1, top_pending: null };
      return reply({ event: { id: decide[1], status: key }, counts });
    }

    if (url === "/api/classify" && method === "POST") {
      const fd = opts.body;
      const hasThermal = fd instanceof FormData && fd.has("thermal");
      const r = hasThermal ? RESULTS.hot : RESULTS.cold;
      return new Promise((res) => setTimeout(() => res({
        ok: true, status: 200,
        json: async () => ({
          result: r,
          event: { ...PENDING[0], id: "preview-cls", ts: Date.now() / 1000,
                   camera_name: "Manual upload", location: "",
                   rgb_score: r.rgb, thermal_score: r.thermal,
                   fused_score: r.fused, verdict: r.verdict, label: r.label,
                   severity: r.severity, reason: r.reason,
                   thermal_image: hasThermal ? PENDING[0].thermal_image : null },
        }),
      }), 650));
    }

    if (url === "/api/cameras/test" && method === "POST") {
      return new Promise((res) => setTimeout(() => res({
        ok: true, status: 200,
        json: async () => ({ ok: true, latency_ms: 214, width: 640, height: 480,
                             source: "http://192.168.1.7:8080/video" }),
      }), 800));
    }

    if (url.startsWith("/api/cameras") || url.startsWith("/api/settings")) {
      return reply({ ok: true, cameras: CAMS, camera: CAMS[0],
                     fusion: __CFG__, removed: true });
    }
    return reply({});
  };

  // core.js builds captured-frame thumbnails as /static/<path>, which has
  // nothing to resolve against offline. Swap them for the inlined samples.
  const SHOTS = __SHOTS__;
  const fixShots = (root) => {
    root.querySelectorAll('img[src^="/static/snapshots/"]').forEach((img) => {
      const key = img.getAttribute("src").replace("/static/", "");
      if (SHOTS[key]) img.src = SHOTS[key];
    });
  };
  new MutationObserver((muts) => muts.forEach((m) =>
    m.addedNodes.forEach((n) => n.nodeType === 1 && fixShots(n))
  )).observe(document.documentElement, { childList: true, subtree: true });
  addEventListener("DOMContentLoaded", () => fixShots(document));

  window.confirm = () => true;
})();
"""


def mock_js():
    return (MOCK_JS
            .replace("__CAMS__", json.dumps(CAMS))
            .replace("__COUNTS__", json.dumps(COUNTS))
            .replace("__PENDING__", json.dumps(PENDING))
            .replace("__DEMO__", json.dumps(DEMO_RESULT))
            .replace("__COLD__", json.dumps(COLD_RESULT))
            .replace("__SHOTS__", json.dumps(SNAP_MAP))
            .replace("__CFG__", json.dumps(FusionConfig().to_dict())))


BANNER = """
<div style="position:fixed;left:50%;bottom:14px;transform:translateX(-50%);z-index:120;
     display:flex;gap:10px;align-items:center;padding:8px 14px;border-radius:3px;
     background:#1D2447;border:1px solid #3A4480;font-size:12px;color:#A9B1DC;
     font-family:Archivo,system-ui,sans-serif;max-width:calc(100vw - 28px)">
  <strong style="color:#E7EAFB;font-weight:650">Preview</strong>
  <span>Sample frames and a stubbed backend. Buttons, the alarm and the
  classifier all work.</span>
  <button onclick="this.parentElement.remove()" style="background:none;border:0;
     color:#737CAE;cursor:pointer;font:inherit;padding:0 2px">dismiss</button>
</div>
"""

PAGES = [("live.html", dict(cameras=CAMS)),
         ("classifier.html", {}),
         ("review.html", dict(events=PENDING)),
         ("history.html", dict(events=ALL_EVENTS, stats=STATS)),
         ("cameras.html", dict(cameras=CAMS))]

BASE_CTX = dict(nav=NAV, cv_ok=True, cfg=FusionConfig().to_dict(),
                verdict_meta=VERDICT_META, counts=COUNTS)

CSS_LINK = '<link rel="stylesheet" href="__STATIC__/css/app.css">'
JS_TAG = '<script src="__STATIC__/js/core.js"></script>'

for tpl, extra in PAGES:
    slug = tpl.replace(".html", "")
    html = env.get_template(tpl).render(**{**BASE_CTX, **extra, "active": slug})

    # Plain string replacement, not re.sub: the shim contains backslash escapes
    # that re would try to interpret in the replacement text.
    assert CSS_LINK in html and JS_TAG in html, f"{slug}: asset tags not found"
    html = html.replace(CSS_LINK, f"<style>\n{CSS}\n</style>")
    html = html.replace(JS_TAG, f"<script>{mock_js()}</script>\n<script>{CORE_JS}</script>")

    # live feeds -> sample frames
    html = re.sub(r'src="/stream/([^"]+)"',
                  lambda m: f'src="{URI.get(m.group(1), URI["cam2"])}"', html)

    html = html.replace("</body>", BANNER + "\n</body>")
    html = html.replace("__STATIC__/", "")

    path = os.path.join(OUT, f"{slug}.html")
    with open(path, "w") as f:
        f.write(html)
    assert "__STATIC__" not in html and "/api/" in html
    print(f"  {slug + '.html':22} {len(html) / 1024:7.0f} KB")

print(f"\nPreview written to {OUT}")
print(f"Demo pair scored: {DEMO_RESULT['verdict']} @ {DEMO_RESULT['fused']:.2f} "
      f"(rgb {DEMO_RESULT['rgb']:.2f}, thermal {DEMO_RESULT['thermal']:.2f})")
print(f"Same visible image, cold thermal: {COLD_RESULT['verdict']} "
      f"@ {COLD_RESULT['fused']:.2f}")
