"""
SQLite store: cameras, detection events, and dispatch decisions.

One file, no server to install, survives a restart -- which matters because the
review queue must not lose a pending alert when you stop the app.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import List, Optional

from .cameras import CameraSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id        TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    source    TEXT NOT NULL,
    location  TEXT DEFAULT '',
    modality  TEXT DEFAULT 'rgb',
    pair_id   TEXT,
    enabled   INTEGER DEFAULT 1,
    added_at  REAL
);

CREATE TABLE IF NOT EXISTS events (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    camera_id     TEXT,
    camera_name   TEXT,
    location      TEXT DEFAULT '',
    source        TEXT DEFAULT 'live',   -- live | upload | test
    verdict       TEXT,
    label         TEXT,
    severity      INTEGER DEFAULT 0,
    rgb_score     REAL,
    thermal_score REAL,
    fused_score   REAL,
    reason        TEXT,
    steps_json    TEXT,
    matrix_cell   TEXT,
    rgb_image     TEXT,
    thermal_image TEXT,
    status        TEXT DEFAULT 'pending', -- pending | dispatched | dismissed
    decided_at    REAL,
    note          TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Store:
    def __init__(self, path="firewatch.db"):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ cameras
    def list_cameras(self) -> List[CameraSpec]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT * FROM cameras ORDER BY added_at").fetchall()
        return [CameraSpec(id=r["id"], name=r["name"], source=r["source"],
                           location=r["location"] or "", modality=r["modality"] or "rgb",
                           pair_id=r["pair_id"], enabled=bool(r["enabled"]))
                for r in rows]

    def add_camera(self, name, source, location="", modality="rgb",
                   pair_id=None) -> CameraSpec:
        cam_id = uuid.uuid4().hex[:8]
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO cameras (id,name,source,location,modality,pair_id,enabled,added_at)"
                " VALUES (?,?,?,?,?,?,1,?)",
                (cam_id, name, str(source), location, modality, pair_id, time.time()))
        return CameraSpec(id=cam_id, name=name, source=str(source), location=location,
                          modality=modality, pair_id=pair_id, enabled=True)

    def remove_camera(self, cam_id):
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM cameras WHERE id=?", (cam_id,))
            c.execute("UPDATE cameras SET pair_id=NULL WHERE pair_id=?", (cam_id,))

    EDITABLE = ("name", "source", "location", "modality", "pair_id")

    def update_camera(self, cam_id, **fields) -> Optional[CameraSpec]:
        """Partial update. Unknown keys are ignored; missing ones are left alone."""
        sets = {k: fields[k] for k in self.EDITABLE if k in fields}
        if "source" in sets:
            sets["source"] = str(sets["source"])
        if not sets:
            return next((c for c in self.list_cameras() if c.id == cam_id), None)

        clause = ", ".join(f"{k}=?" for k in sets)
        with self._lock, self._conn() as c:
            cur = c.execute(f"UPDATE cameras SET {clause} WHERE id=?",
                            (*sets.values(), cam_id))
            if not cur.rowcount:
                return None
            # A camera cannot be its own thermal twin, and a thermal camera has
            # no use for a twin of its own.
            c.execute("UPDATE cameras SET pair_id=NULL WHERE id=? AND"
                      " (pair_id=id OR modality='thermal')", (cam_id,))
            # If this one stopped being thermal, nothing may still pair to it.
            if sets.get("modality") == "rgb":
                c.execute("UPDATE cameras SET pair_id=NULL WHERE pair_id=?", (cam_id,))
        return next((c for c in self.list_cameras() if c.id == cam_id), None)

    def set_camera_enabled(self, cam_id, enabled: bool):
        with self._lock, self._conn() as c:
            c.execute("UPDATE cameras SET enabled=? WHERE id=?",
                      (1 if enabled else 0, cam_id))

    # ------------------------------------------------------------- events
    def add_event(self, result: dict, camera_id=None, camera_name="Manual upload",
                  location="", source="live", rgb_image=None,
                  thermal_image=None) -> dict:
        ev_id = uuid.uuid4().hex[:10]
        row = (
            ev_id, time.time(), camera_id, camera_name, location, source,
            result.get("verdict"), result.get("label"), result.get("severity", 0),
            result.get("rgb"), result.get("thermal"), result.get("fused"),
            result.get("reason"), json.dumps(result.get("steps", [])),
            result.get("matrix_cell"), rgb_image, thermal_image, "pending", None, "",
        )
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO events (id,ts,camera_id,camera_name,location,source,verdict,"
                "label,severity,rgb_score,thermal_score,fused_score,reason,steps_json,"
                "matrix_cell,rgb_image,thermal_image,status,decided_at,note)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        return self.get_event(ev_id)

    def get_event(self, ev_id) -> Optional[dict]:
        with self._lock, self._conn() as c:
            r = c.execute("SELECT * FROM events WHERE id=?", (ev_id,)).fetchone()
        return _event_dict(r) if r else None

    def list_events(self, status=None, limit=200) -> List[dict]:
        q = "SELECT * FROM events"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with self._lock, self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [_event_dict(r) for r in rows]

    def decide(self, ev_id, status, note="") -> Optional[dict]:
        if status not in ("dispatched", "dismissed", "pending"):
            return None
        with self._lock, self._conn() as c:
            c.execute("UPDATE events SET status=?, decided_at=?, note=? WHERE id=?",
                      (status, time.time(), note or "", ev_id))
        return self.get_event(ev_id)

    def counts(self) -> dict:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM events GROUP BY status").fetchall()
            top = c.execute(
                "SELECT * FROM events WHERE status='pending' ORDER BY severity DESC, ts DESC LIMIT 1"
            ).fetchone()
        out = {"pending": 0, "dispatched": 0, "dismissed": 0}
        for r in rows:
            out[r["status"]] = r["n"]
        out["total"] = sum(v for k, v in out.items() if k != "total")
        out["top_pending"] = _event_dict(top) if top else None
        return out

    def stats(self, days=7) -> dict:
        since = time.time() - days * 86400
        with self._lock, self._conn() as c:
            by_day = c.execute(
                "SELECT strftime('%Y-%m-%d', ts, 'unixepoch', 'localtime') d,"
                " COUNT(*) n,"
                " SUM(CASE WHEN status='dispatched' THEN 1 ELSE 0 END) dispatched,"
                " SUM(CASE WHEN status='dismissed' THEN 1 ELSE 0 END) dismissed"
                " FROM events WHERE ts>=? GROUP BY d ORDER BY d", (since,)).fetchall()
            by_cam = c.execute(
                "SELECT COALESCE(camera_name,'Manual upload') cam, COUNT(*) n"
                " FROM events WHERE ts>=? GROUP BY cam ORDER BY n DESC LIMIT 8",
                (since,)).fetchall()
            by_verdict = c.execute(
                "SELECT verdict, COUNT(*) n FROM events WHERE ts>=? GROUP BY verdict",
                (since,)).fetchall()
        return {
            "by_day": [dict(r) for r in by_day],
            "by_camera": [dict(r) for r in by_cam],
            "by_verdict": [dict(r) for r in by_verdict],
        }

    # ----------------------------------------------------------- settings
    def get_setting(self, key, default=None):
        with self._lock, self._conn() as c:
            r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if not r:
            return default
        try:
            return json.loads(r["value"])
        except Exception:
            return default

    def set_setting(self, key, value):
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO settings (key,value) VALUES (?,?)"
                      " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value)))


def _event_dict(r) -> dict:
    d = dict(r)
    try:
        d["steps"] = json.loads(d.pop("steps_json") or "[]")
    except Exception:
        d["steps"] = []
        d.pop("steps_json", None)
    d["age_s"] = max(0, int(time.time() - (d["ts"] or 0)))
    return d
