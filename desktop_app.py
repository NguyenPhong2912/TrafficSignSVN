import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab, ImageTk
from tkinter import filedialog, messagebox, ttk


BASE_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import inference_backend as backend


WINDOW_TITLE = "Traffic Sign VN Desktop"
DISPLAY_MAX_DIM = 1280
UI_REFRESH_MS = 33
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
CAMERA_SCAN_LIMIT = 10

CAMERA_BACKENDS = {
    "Auto": None,
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
}
CAMERA_SCAN_BACKENDS = ("MSMF", "Auto")
CAMERA_AUTO_OPEN_ORDER = ("MSMF", "Auto")


def resize_to_max_dim(frame, max_dim: int):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame.copy()
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def frame_looks_valid(frame) -> bool:
    if frame is None or frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    std_val = float(gray.std())
    return mean_val > 3.0 or std_val > 2.0


def open_camera_with_backend(index: int, backend_name: str):
    backend_id = CAMERA_BACKENDS[backend_name]
    if backend_id is None:
        camera = cv2.VideoCapture(index)
    else:
        camera = cv2.VideoCapture(index, backend_id)
    return camera


def probe_camera_source(index: int, backend_name: str, width: int = DEFAULT_CAMERA_WIDTH, height: int = DEFAULT_CAMERA_HEIGHT):
    camera = open_camera_with_backend(index, backend_name)
    if not camera.isOpened():
        return None

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    valid_frame = None
    for _ in range(20):
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.03)
            continue
        if frame_looks_valid(frame):
            valid_frame = frame
            break
        time.sleep(0.03)

    if valid_frame is None:
        camera.release()
        return None

    return camera, valid_frame


def probe_camera_with_fallback(index: int, backend_name: str):
    ordered_backends = (backend_name,) if backend_name != "Auto" else CAMERA_AUTO_OPEN_ORDER
    tried = []
    for candidate_backend in ordered_backends:
        tried.append(candidate_backend)
        probed = probe_camera_source(index, candidate_backend)
        if probed is not None:
            camera, frame = probed
            return camera, frame, candidate_backend, tried
    return None, None, None, tried


def strip_result_for_json(result: dict[str, Any]) -> dict[str, Any]:
    clean = {"signs": [], "plates": []}
    for sign in result.get("signs", []):
        clean["signs"].append(dict(sign))
    for plate in result.get("plates", []):
        item = dict(plate)
        item.pop("crop_b64", None)
        clean["plates"].append(item)
    return clean


def draw_result_overlay(frame, result: dict[str, Any]):
    rendered = frame.copy()

    for sign in result.get("signs", []):
        x1, y1, x2, y2 = sign["bbox"]
        label = f"{sign['label']} {int(sign['score'] * 100)}%"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (50, 205, 50), 2)
        cv2.rectangle(rendered, (x1, max(0, y1 - 24)), (x1 + max(120, len(label) * 8), y1), (30, 90, 30), -1)
        cv2.putText(rendered, label, (x1 + 4, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 255, 90), 2)

    for plate in result.get("plates", []):
        x1, y1, x2, y2 = plate["bbox"]
        text = plate.get("plate_text") or "Plate"
        label = f"{text} {int(plate['detect_conf'] * 100)}%"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (80, 140, 255), 2)
        cv2.rectangle(rendered, (x1, max(0, y1 - 24)), (x1 + max(120, len(label) * 8), y1), (40, 70, 130), -1)
        cv2.putText(rendered, label, (x1 + 4, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 2)

    return rendered


def summarize_result(result: dict[str, Any]) -> str:
    lines = []
    signs = result.get("signs", [])
    plates = result.get("plates", [])
    lines.append(f"Signs: {len(signs)}")
    for sign in signs[:10]:
        lines.append(f"  - {sign['label']} ({sign['score']:.2f})")
    if len(signs) > 10:
        lines.append(f"  ... {len(signs) - 10} more")

    lines.append("")
    lines.append(f"Plates: {len(plates)}")
    for plate in plates[:10]:
        text = plate.get("plate_text") or "unknown"
        lines.append(f"  - {text} ({plate['detect_conf']:.2f})")
    if len(plates) > 10:
        lines.append(f"  ... {len(plates) - 10} more")
    return "\n".join(lines)


class DesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry("1500x920")
        self.minsize(1180, 760)
        self.configure(bg="#101418")

        self.models_ready = False
        self.model_error = None
        self.camera_running = False
        self.preview_enabled = True
        self.static_image_mode = False

        self.capture_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_preview_frame = None
        self.latest_preview_result = {"signs": [], "plates": []}
        self.latest_still_frame = None
        self.latest_still_result = {"signs": [], "plates": []}
        self.last_preview_duration = 0.0
        self.last_snapshot_path = None
        self.last_frame_ts = 0.0
        self.last_preview_ts = 0.0
        self.last_dense_preview_ts = 0.0
        self.available_sources: list[str] = []
        self.current_camera_label = None
        self.capture_mode = None
        self.screen_region: tuple[int, int, int, int] | None = None

        self.camera = None
        self.capture_thread = None
        self.preview_thread = None
        self.stop_event = threading.Event()

        self.preview_image_tk = None

        self.status_var = tk.StringVar(value="Loading models...")
        self.model_var = tk.StringVar(value="Waiting...")
        self.camera_var = tk.StringVar(value="Camera stopped")
        self.preview_var = tk.StringVar(value="Preview: on")
        self.perf_var = tk.StringVar(value="FPS display: 0.0 | FPS infer: 0.0")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self._load_models_worker, daemon=True).start()
        self.after(UI_REFRESH_MS, self._ui_tick)

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101418")
        style.configure("TLabel", background="#101418", foreground="#eef3f8")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", foreground="#9cb0c3")
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#101418", foreground="#eef3f8")

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=4)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Traffic Sign VN Desktop", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.model_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(top, textvariable=self.status_var, style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        left = ttk.Frame(root)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        controls = ttk.Frame(left)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(controls, text="Camera").grid(row=0, column=0, padx=(0, 6))
        self.camera_source_var = tk.StringVar(value="0 | MSMF")
        self.camera_source_combo = ttk.Combobox(
            controls,
            width=18,
            textvariable=self.camera_source_var,
            values=["0 | MSMF", "0 | Auto"],
            state="readonly",
        )
        self.camera_source_combo.grid(row=0, column=1, padx=(0, 8))
        self.refresh_btn = ttk.Button(controls, text="Scan", command=self.refresh_camera_sources, state="disabled")
        self.refresh_btn.grid(row=0, column=2, padx=(0, 10))
        self.region_btn = ttk.Button(controls, text="Select Region", command=self.select_screen_region, state="disabled")
        self.region_btn.grid(row=0, column=3, padx=(0, 10))

        self.start_btn = ttk.Button(controls, text="Start Camera", command=self.start_camera, state="disabled")
        self.start_btn.grid(row=0, column=4, padx=(0, 6))
        self.stop_btn = ttk.Button(controls, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=5, padx=(0, 12))

        self.preview_btn = ttk.Button(controls, text="Pause Preview", command=self.toggle_preview, state="disabled")
        self.preview_btn.grid(row=0, column=6, padx=(0, 6))
        self.snapshot_btn = ttk.Button(controls, text="Full Snapshot", command=self.snapshot_full, state="disabled")
        self.snapshot_btn.grid(row=0, column=7, padx=(0, 6))
        self.open_btn = ttk.Button(controls, text="Open Image", command=self.open_image, state="disabled")
        self.open_btn.grid(row=0, column=8, padx=(0, 6))

        ttk.Label(controls, textvariable=self.camera_var, style="Muted.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.preview_var, style="Muted.TLabel").grid(row=1, column=6, columnspan=1, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.perf_var, style="Muted.TLabel").grid(row=1, column=7, columnspan=2, sticky="e", pady=(8, 0))

        self.preview_label = ttk.Label(left, anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        right = ttk.Frame(root)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Detection Results", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.result_text = tk.Text(
            right,
            wrap="word",
            bg="#0a0d10",
            fg="#d9e2ec",
            insertbackground="#d9e2ec",
            relief="flat",
            font=("Consolas", 11),
            padx=10,
            pady=10,
        )
        self.result_text.grid(row=1, column=0, sticky="nsew")

        bottom = ttk.Frame(right)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.last_save_var = tk.StringVar(value="No capture saved yet")
        ttk.Label(bottom, textvariable=self.last_save_var, style="Muted.TLabel").pack(anchor="w")

    def _load_models_worker(self):
        try:
            backend.load_models()
            self.models_ready = backend.models["loaded"]
            self.model_error = backend.models["error"]
        except Exception as exc:
            self.models_ready = False
            self.model_error = str(exc)

        def finish():
            if self.models_ready:
                self.model_var.set(
                    f"Device: {backend.DEVICE.upper()} | Sign: {'yes' if backend.models['yolo_sign'] else 'no'}"
                    f" | Plate: {'yes' if backend.models['yolo_plate'] else 'no'}"
                    f" | CLIP: {'yes' if backend.models['clip_clf'] else 'no'}"
                    f" | OCR: {'yes' if backend.models['ocr_reader'] else 'no'}"
                )
                self.status_var.set("Models loaded. Start the Windows camera.")
                self.start_btn.config(state="normal")
                self.preview_btn.config(state="normal")
                self.snapshot_btn.config(state="normal")
                self.open_btn.config(state="normal")
                self.refresh_btn.config(state="normal")
                self.region_btn.config(state="normal")
                self.refresh_camera_sources()
            else:
                self.model_var.set("Model load failed")
                self.status_var.set(self.model_error or "Unknown model load error")
                messagebox.showerror("Model load failed", self.model_error or "Unknown error")

        self.after(0, finish)

    def refresh_camera_sources(self):
        if self.camera_running:
            self.status_var.set("Stop current camera before scanning.")
            return

        self.status_var.set("Scanning cameras...")
        self.update_idletasks()

        found = []
        for index in range(CAMERA_SCAN_LIMIT):
            for backend_name in CAMERA_SCAN_BACKENDS:
                probed = probe_camera_source(index, backend_name, width=640, height=480)
                if probed is None:
                    continue
                camera, frame = probed
                try:
                    frame_h, frame_w = frame.shape[:2]
                    found.append(f"{index} | {backend_name} | {frame_w}x{frame_h}")
                finally:
                    camera.release()

        if not found:
            found = ["0 | MSMF", "0 | Auto"]
            self.status_var.set("No validated camera stream found. Use Select Region for Camo, or fix Camo preview first, then Scan again.")
        else:
            self.status_var.set(f"Found {len(found)} camera source(s). Select Region is available for Camo/virtual cameras.")

        self.available_sources = found
        self.camera_source_combo["values"] = found
        self.camera_source_var.set(found[0])

    def start_camera(self):
        if not self.models_ready or self.camera_running:
            return

        try:
            source_text = self.camera_source_var.get().strip()
            source_parts = [part.strip() for part in source_text.split("|")]
            camera_index = int(source_parts[0])
            backend_name = source_parts[1] if len(source_parts) > 1 and source_parts[1] in CAMERA_BACKENDS else "Auto"
        except Exception:
            messagebox.showerror("Invalid camera", "Select a camera source from the Scan list.")
            return

        camera, first_frame, actual_backend, tried_backends = probe_camera_with_fallback(camera_index, backend_name)
        if camera is None:
            messagebox.showerror(
                "Camera error",
                f"Cannot read a valid frame from camera {camera_index}.\n"
                f"Tried: {', '.join(tried_backends)}\n\n"
                "If you use Camo, make sure Camo Studio already shows a live image. "
                "If Camo itself is black, this app cannot receive a valid frame either.",
            )
            return

        self.camera = camera
        self.camera_running = True
        self.static_image_mode = False
        self.stop_event.clear()
        self.capture_mode = "camera"
        self.screen_region = None
        self.current_camera_label = f"{camera_index} via {actual_backend}"
        with self.capture_lock:
            self.latest_raw_frame = first_frame
            self.last_frame_ts = time.perf_counter()
        self.camera_var.set(f"Camera {self.current_camera_label} running")
        self.status_var.set("Live preview enabled")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.capture_thread.start()
        self.preview_thread.start()

    def stop_camera(self):
        self.camera_running = False
        self.stop_event.set()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        self.camera = None
        self.current_camera_label = None
        self.capture_mode = None
        self.start_btn.config(state="normal" if self.models_ready else "disabled")
        self.stop_btn.config(state="disabled")
        self.camera_var.set("Camera stopped")

    def select_screen_region(self):
        if not self.models_ready:
            return
        if self.camera_running:
            self.stop_camera()

        self.status_var.set("Drag to select a screen region for realtime capture.")
        self.after(50, self._open_region_selector)

    def _open_region_selector(self):
        selector = tk.Toplevel(self)
        selector.title("Select Capture Region")
        selector.attributes("-fullscreen", True)
        selector.attributes("-topmost", True)
        try:
            selector.attributes("-alpha", 0.22)
        except Exception:
            pass
        selector.configure(bg="black")
        selector.focus_force()
        selector.grab_set()

        canvas = tk.Canvas(selector, bg="black", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True)
        canvas.create_text(
            24,
            24,
            anchor="nw",
            fill="#ffffff",
            font=("Segoe UI", 16, "bold"),
            text="Drag to select the Camo preview region. Press Esc to cancel.",
        )

        selection: dict[str, Any] = {"start": None, "rect": None}

        def on_press(event):
            selection["start"] = (event.x_root, event.y_root)
            if selection["rect"] is not None:
                canvas.delete(selection["rect"])
            selection["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#55d6ff", width=3)

        def on_drag(event):
            if selection["start"] is None or selection["rect"] is None:
                return
            start_x, start_y = selection["start"]
            canvas.coords(selection["rect"], start_x, start_y, event.x_root, event.y_root)

        def finish(region):
            try:
                selector.grab_release()
            except Exception:
                pass
            selector.destroy()
            if region is None:
                self.status_var.set("Screen region selection cancelled.")
                return
            self.start_screen_region(region)

        def on_release(event):
            if selection["start"] is None:
                finish(None)
                return
            start_x, start_y = selection["start"]
            x1, y1 = min(start_x, event.x_root), min(start_y, event.y_root)
            x2, y2 = max(start_x, event.x_root), max(start_y, event.y_root)
            if x2 - x1 < 40 or y2 - y1 < 40:
                messagebox.showinfo("Region too small", "Select a larger region to capture.")
                finish(None)
                return
            finish((int(x1), int(y1), int(x2), int(y2)))

        def on_escape(_event=None):
            finish(None)

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        selector.bind("<Escape>", on_escape)

    def start_screen_region(self, region: tuple[int, int, int, int]):
        frame = self._grab_screen_region_frame(region)
        if frame is None or not frame_looks_valid(frame):
            messagebox.showerror(
                "Screen capture error",
                "Cannot capture a valid frame from the selected region.\n"
                "Make sure the Camo preview is visible and not fully black, then select the region again.",
            )
            self.status_var.set("Screen region did not return a valid frame.")
            return

        self.camera_running = True
        self.static_image_mode = False
        self.stop_event.clear()
        self.capture_mode = "screen"
        self.screen_region = region
        self.current_camera_label = f"screen {region[2] - region[0]}x{region[3] - region[1]}"
        with self.capture_lock:
            self.latest_raw_frame = frame
            self.last_frame_ts = time.perf_counter()
        self.camera_var.set(f"Screen region {self.current_camera_label} running")
        self.status_var.set("Screen region preview enabled")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.capture_thread.start()
        self.preview_thread.start()

    def _grab_screen_region_frame(self, region: tuple[int, int, int, int] | None = None):
        bbox = region or self.screen_region
        if bbox is None:
            return None
        try:
            image = ImageGrab.grab(bbox=bbox, all_screens=True)
        except Exception:
            return None
        frame = np.array(image)
        if frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def toggle_preview(self):
        self.preview_enabled = not self.preview_enabled
        self.preview_var.set(f"Preview: {'on' if self.preview_enabled else 'paused'}")
        self.preview_btn.config(text="Pause Preview" if self.preview_enabled else "Resume Preview")
        if self.preview_enabled:
            self.status_var.set("Preview detect running")
        else:
            self.status_var.set("Preview paused")

    def _capture_loop(self):
        while self.camera_running:
            if self.capture_mode == "screen":
                frame = self._grab_screen_region_frame()
                if frame is None or not frame_looks_valid(frame):
                    time.sleep(0.02)
                    continue
            else:
                if self.camera is None:
                    break
                ok, frame = self.camera.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if not frame_looks_valid(frame):
                    time.sleep(0.02)
                    continue
            with self.capture_lock:
                self.latest_raw_frame = frame
                self.last_frame_ts = time.perf_counter()

    def _preview_loop(self):
        while self.camera_running:
            if not self.preview_enabled:
                time.sleep(0.03)
                continue

            with self.capture_lock:
                frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            preview_frame = resize_to_max_dim(frame, DISPLAY_MAX_DIM)
            dense_preview = self.capture_mode == "screen"
            now = time.perf_counter()
            run_dense = dense_preview and (now - self.last_dense_preview_ts >= 0.6)
            started = time.perf_counter()
            try:
                result = backend.run_pipeline(
                    preview_frame,
                    enable_clip=False,
                    enable_ocr=False,
                    include_plate_crop=False,
                    preview_mode=True,
                    dense_plate=run_dense,
                )
            except Exception as exc:
                self.after(0, lambda: self.status_var.set(f"Preview error: {exc}"))
                time.sleep(0.1)
                continue

            duration = time.perf_counter() - started
            with self.capture_lock:
                self.latest_preview_frame = preview_frame
                self.latest_preview_result = result
                self.last_preview_duration = duration
                self.last_preview_ts = time.perf_counter()
                if run_dense:
                    self.last_dense_preview_ts = self.last_preview_ts

    def _ui_tick(self):
        frame_to_show = None
        result_to_show = {"signs": [], "plates": []}

        with self.capture_lock:
            if self.static_image_mode and self.latest_still_frame is not None:
                frame_to_show = self.latest_still_frame.copy()
                result_to_show = self.latest_still_result
            elif self.latest_preview_frame is not None:
                frame_to_show = self.latest_preview_frame.copy()
                result_to_show = self.latest_preview_result
            elif self.latest_raw_frame is not None:
                frame_to_show = resize_to_max_dim(self.latest_raw_frame, DISPLAY_MAX_DIM)

            preview_fps = 0.0 if self.last_preview_duration <= 0 else 1.0 / self.last_preview_duration

        if frame_to_show is not None:
            rendered = draw_result_overlay(frame_to_show, result_to_show)
            rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rendered_rgb)
            image.thumbnail((980, 700))
            self.preview_image_tk = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.preview_image_tk)
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", summarize_result(result_to_show))

        display_age = max(0.0, time.perf_counter() - self.last_frame_ts) if self.last_frame_ts else 0.0
        display_fps = 0.0 if display_age <= 0 else min(60.0, 1.0 / max(display_age, 1e-3))
        self.perf_var.set(f"FPS display: {display_fps:0.1f} | FPS infer: {preview_fps:0.1f}")

        self.after(UI_REFRESH_MS, self._ui_tick)

    def snapshot_full(self):
        if not self.models_ready:
            return

        with self.capture_lock:
            source = None
            if self.static_image_mode and self.latest_still_frame is not None:
                source = self.latest_still_frame.copy()
            elif self.latest_raw_frame is not None:
                source = self.latest_raw_frame.copy()

        if source is None:
            messagebox.showinfo("No frame", "No camera frame or image available.")
            return

        self.snapshot_btn.config(state="disabled")
        self.status_var.set("Running full pipeline...")
        threading.Thread(target=self._run_full_snapshot_worker, args=(source,), daemon=True).start()

    def _run_full_snapshot_worker(self, frame):
        started = time.perf_counter()
        result = backend.run_pipeline(
            frame,
            enable_clip=True,
            enable_ocr=True,
            include_plate_crop=True,
            preview_mode=False,
            dense_plate=self.capture_mode == "screen",
        )
        duration = time.perf_counter() - started

        saved_path = self._save_capture(frame, result)
        display_frame = resize_to_max_dim(frame, DISPLAY_MAX_DIM)

        def finish():
            with self.capture_lock:
                if self.camera_running:
                    self.latest_preview_frame = display_frame
                    self.latest_preview_result = result
                else:
                    self.latest_still_frame = display_frame
                    self.latest_still_result = result
                    self.static_image_mode = True
            self.last_snapshot_path = saved_path
            self.last_save_var.set(f"Saved: {saved_path}")
            self.status_var.set(f"Full snapshot done in {duration:0.2f}s")
            self.snapshot_btn.config(state="normal")

        self.after(0, finish)

    def open_image(self):
        if not self.models_ready:
            return

        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Image error", f"Cannot read image: {path}")
            return

        self.stop_camera()
        self.snapshot_btn.config(state="disabled")
        self.status_var.set("Running full pipeline on image...")
        threading.Thread(target=self._run_full_snapshot_worker, args=(image,), daemon=True).start()

    def _save_capture(self, frame, result: dict[str, Any]) -> Path:
        capture_id = uuid.uuid4().hex[:8]
        ts = datetime.now(timezone.utc).isoformat()
        image_path = CAPTURES_DIR / f"{capture_id}.jpg"
        overlay_path = CAPTURES_DIR / f"{capture_id}_overlay.jpg"
        meta_path = CAPTURES_DIR / f"{capture_id}.json"

        cv2.imwrite(str(image_path), frame)
        cv2.imwrite(str(overlay_path), draw_result_overlay(frame, result))

        meta = {
            "id": capture_id,
            "timestamp": ts,
            "source": "desktop-screen" if self.capture_mode == "screen" else "desktop-camera",
            "image_file": image_path.name,
            "overlay_file": overlay_path.name,
            "result": strip_result_for_json(result),
            "device": backend.DEVICE,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def on_close(self):
        self.stop_camera()
        self.destroy()


def run_self_check():
    backend.load_models()
    print(
        json.dumps(
            {
                "loaded": backend.models["loaded"],
                "ready_full_flow": backend.models["ready_full_flow"],
                "device": backend.DEVICE,
                "paths": backend.models["paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    if "--self-check" in sys.argv:
        run_self_check()
        return

    app = DesktopApp()
    app.mainloop()


if __name__ == "__main__":
    main()
