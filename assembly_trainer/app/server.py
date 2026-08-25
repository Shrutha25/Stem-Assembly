"""FastAPI server (build order item 10): trainee UI, trainer dashboard,
live MJPEG feed, and a WebSocket pushing pipeline state.

Runs the `Pipeline` (camera/mock -> detector -> state machine -> escalation)
on a background thread at `app.inference_interval_ms`, and exposes its
output to the UI through a small thread-safe `AppState`.

Usage:
    python -m assembly_trainer.app.server --config config/default_config.yaml
    python -m assembly_trainer.app.server --mock                 # no camera/model needed
    python -m assembly_trainer.app.server --mock --no-camera     # no hardware at all
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..camera import ThreadedCamera
from ..config import Config, load_config
from ..detector import Detector, YoloByteTrackDetector
from ..mock_detector import MockDetector, build_full_walkthrough_script
from ..pipeline import Pipeline, PipelineResult
from ..pose_detector import PoseCornerDetector

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
REPO_ROOT = APP_DIR.parent.parent


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {"status": "starting"}

    def update(self, jpeg: bytes, state: dict) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._state = state

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self._lock:
            return self._jpeg, dict(self._state)


def _placeholder_frame(width: int, height: int, text: str) -> np.ndarray:
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(frame, text, (40, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    return frame


def _annotate(frame: np.ndarray, result: PipelineResult, detections, is_mock: bool) -> np.ndarray:
    out = frame.copy()
    fr = result.frame_result
    if is_mock:
        # Impossible-to-miss watermark: MockDetector's scripted walkthrough
        # advances on a timer regardless of what's actually in frame — the
        # real camera feed is still shown underneath, so without this a demo
        # run is indistinguishable from genuine live detection (this exact
        # confusion is what prompted adding it).
        cv2.rectangle(out, (0, out.shape[0] - 46), (out.shape[1], out.shape[0]), (0, 0, 200), -1)
        cv2.putText(out, "DEMO MODE - no trained model loaded, this is a scripted walkthrough (not real detection)",
                    (16, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if fr.roi is not None:
        x1, y1, x2, y2 = (int(v) for v in fr.roi)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, "placement zone (only detections in here count)", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{d.class_name} {d.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    banner = f"Step {fr.step_id}: {fr.step_name}" if not fr.completed else "Assembly complete"
    cv2.putText(out, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if not fr.completed:
        cv2.putText(out, f"tier={result.tier.name} elapsed={result.elapsed_on_step:.0f}s",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return out


_DIAGNOSIS_MESSAGES = {
    "satisfied": None,
    "not_present": "not detected yet",
    "uncertain": "not sure yet — hold steady / move closer",
    "wrong_orientation": "wrong orientation — flip/rotate it",
    "wrong_part": "wrong part detected here",
    "wrong_hole": "wrong hole — move it to the correct hole",
}


def _diagnosis_message(diag) -> str | None:
    base = _DIAGNOSIS_MESSAGES.get(diag.reason.value)
    if diag.reason.value == "wrong_part" and diag.detail:
        return f"wrong part detected here ({diag.detail.replace('_', ' ')})"
    if diag.reason.value == "wrong_hole" and diag.detail:
        return f"wrong hole — currently in {diag.detail.replace('_', ' ')}, move it to the correct hole"
    return base


def _state_dict(cfg: Config, result: PipelineResult, is_mock: bool) -> dict:
    fr = result.frame_result
    return {
        "status": "running",
        "is_mock": is_mock,
        "step_id": fr.step_id,
        "step_name": fr.step_name,
        "total_steps": len(cfg.steps),
        "completed": fr.completed,
        "advanced": fr.advanced,
        "progress_ratio": fr.progress_ratio,
        "class_status": fr.class_status,
        "class_diagnosis": {
            cls: {"reason": diag.reason.value, "message": _diagnosis_message(diag)}
            for cls, diag in fr.class_diagnosis.items()
        },
        "sub_step_progress": list(fr.sub_step_progress) if fr.sub_step_progress else None,
        "waiting_for_reset": fr.waiting_for_reset,
        "tier": result.tier.name,
        "elapsed_on_step": round(result.elapsed_on_step, 1),
        "reference_image": _step_asset(cfg, fr.step_id, "reference_image"),
        "reference_video": _step_asset(cfg, fr.step_id, "reference_video"),
    }


def _step_asset(cfg: Config, step_id: int | None, field: str) -> str | None:
    if step_id is None:
        return None
    step = cfg.get_step(step_id)
    value = getattr(step, field)
    return f"/assets/{Path(value).relative_to('assets').as_posix()}" if value else None


def build_detector(cfg: Config, args: argparse.Namespace, frame_w: int, frame_h: int) -> tuple[Detector, bool]:
    """Returns (detector, is_mock)."""
    if args.mock:
        script = build_full_walkthrough_script(cfg, frame_w, frame_h, hold_seconds=args.mock_hold_seconds)
        return MockDetector(script, loop=True), True

    model_path = Path(args.model or cfg.model_path)
    if not model_path.exists():
        print(f"[server] WARNING: model_path '{model_path}' does not exist (no trained model yet). "
              f"Falling back to --mock so the server/UI plumbing is still demonstrable.")
        script = build_full_walkthrough_script(cfg, frame_w, frame_h, hold_seconds=args.mock_hold_seconds)
        return MockDetector(script, loop=True), True

    try:
        return YoloByteTrackDetector(str(model_path), cfg.classes, imgsz=cfg.app.imgsz, device=args.device), False
    except RuntimeError as exc:
        print(f"[server] WARNING: {exc}\n[server] Falling back to --mock.")
        script = build_full_walkthrough_script(cfg, frame_w, frame_h, hold_seconds=args.mock_hold_seconds)
        return MockDetector(script, loop=True), True


def build_pose_detector(cfg: Config, args: argparse.Namespace) -> PoseCornerDetector | None:
    """None whenever hole-position checking isn't available -- no mock
    fallback (unlike build_detector): this is a purely-optional secondary
    feature, so "just don't run it" is the right degraded behavior rather
    than substituting synthetic data for steps 1/2's real hole check."""
    if args.mock or not cfg.pose_model_path:
        return None
    pose_model_path = Path(cfg.pose_model_path)
    if not pose_model_path.exists():
        print(f"[server] NOTE: pose_model_path '{pose_model_path}' does not exist yet -- "
              f"steps 1/2 hole-position checking disabled, count-only gating still works.")
        return None
    try:
        return PoseCornerDetector(str(pose_model_path), imgsz=cfg.app.imgsz, device=args.device)
    except RuntimeError as exc:
        print(f"[server] NOTE: {exc}\n[server] Hole-position checking disabled.")
        return None


def make_app(args: argparse.Namespace) -> FastAPI:
    cfg = load_config(args.config)
    state = AppState()

    camera: ThreadedCamera | None = None
    frame_w, frame_h = cfg.camera.width, cfg.camera.height
    if not args.no_camera:
        camera = ThreadedCamera(cfg.camera).start()
        frame_w, frame_h = camera.actual_resolution
        print(f"[server] camera started at {frame_w}x{frame_h}")

    detector, is_mock = build_detector(cfg, args, frame_w, frame_h)
    print(f"[server] detector: {'MockDetector (synthetic)' if is_mock else 'YoloByteTrackDetector'}")

    pose_detector = build_pose_detector(cfg, args)
    print(f"[server] pose detector: {'PoseCornerDetector (hole-position checking on)' if pose_detector else 'none (steps 1/2 count-only)'}")

    placeholder = _placeholder_frame(frame_w, frame_h, "no camera (--no-camera)")

    def frame_source():
        if camera is not None:
            frame = camera.get_latest_frame()
            if frame is not None:
                return frame
        return placeholder

    pipeline = Pipeline(cfg, detector, frame_source, pose_detector=pose_detector)
    pipeline_lock = threading.Lock()
    stop_event = threading.Event()

    def loop():
        interval_s = cfg.app.inference_interval_ms / 1000.0
        while not stop_event.is_set():
            t0 = time.monotonic()
            with pipeline_lock:
                result = pipeline.step()
            if result is not None:
                annotated = _annotate(result.frame, result, result.detections, is_mock)
                ok, buf = cv2.imencode(".jpg", annotated)
                if ok:
                    state.update(buf.tobytes(), _state_dict(cfg, result, is_mock))
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval_s - elapsed))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        stop_event.set()
        if camera is not None:
            camera.stop()

    app = FastAPI(title="Assembly Trainer", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/assets", StaticFiles(directory=str(REPO_ROOT / "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/trainer")
    def trainer():
        return FileResponse(str(STATIC_DIR / "trainer.html"))

    @app.get("/api/state")
    def api_state():
        _, s = state.snapshot()
        return JSONResponse(s)

    @app.get("/api/flags")
    def api_flags():
        return JSONResponse([{"step_id": f.step_id, "step_name": f.step_name, "at": f.at} for f in pipeline.flags])

    @app.post("/api/reset")
    def api_reset():
        with pipeline_lock:
            pipeline.reset()
        return JSONResponse({"status": "ok"})

    @app.get("/video_feed")
    def video_feed():
        def gen():
            while True:
                jpeg, _ = state.snapshot()
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(cfg.app.inference_interval_ms / 1000.0)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                _, s = state.snapshot()
                await websocket.send_json(s)
                await asyncio.sleep(cfg.app.inference_interval_ms / 1000.0)
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--model", default=None, help="override config's model_path")
    parser.add_argument("--device", default=None, help="torch device for the real detector, e.g. 'cpu' or '0'")
    parser.add_argument("--mock", action="store_true", help="use the scripted MockDetector instead of a real model")
    parser.add_argument("--mock-hold-seconds", type=float, default=6.0)
    parser.add_argument("--no-camera", action="store_true", help="skip opening a real camera (implies visible frames are placeholders)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = make_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
