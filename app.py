import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename

from video_analyzer import (
    _detect_ratio,
    extract_frame,
    get_aspect_ratio,
    analyze_frame,
    generate_image_magnific,
    generate_video_magnific,
)

REQUEST_TIMEOUT = 600  # seconds


class LongTimeoutHandler(WSGIRequestHandler):
    timeout = REQUEST_TIMEOUT


app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: dict = {}

STEPS = [
    "Extracting frame from video",
    "Detecting aspect ratio",
    "Analyzing frame with GPT-5.5",
    "Generating image (Magnific API)",
    "Generating video (Kling 2.6 Pro)",
    "Pipeline complete",
]


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    video_file = request.files.get("video")
    image_file = request.files.get("image")

    if not video_file and not image_file:
        return jsonify({"error": "No video or image file provided"}), 400

    source_type = "video" if video_file else "image"
    source_file = video_file if source_type == "video" else image_file

    try:
        second = float(request.form.get("second", 0))
    except (ValueError, TypeError):
        second = 0.0

    image_change = request.form.get("image_change") or None
    video_change = request.form.get("video_change") or None
    mode = request.form.get("mode", "full")
    if mode not in ("prompts_only", "image_only", "full"):
        mode = "full"

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(exist_ok=True)

    filename = secure_filename(source_file.filename or ("image.jpg" if source_type == "image" else "video.mp4"))
    source_path = str(upload_path / filename)
    source_file.save(source_path)

    jobs[job_id] = {
        "status": "running",
        "step": 0,
        "step_name": "",
        "mode": mode,
        "source_type": source_type,
        "error": None,
        "frame_ready": False,
        "image_ready": False,
        "video_ready": False,
        "result": None,
    }

    print(f"[{job_id[:8]}] Job created | file={filename} | size={os.path.getsize(source_path)} bytes | mode={mode} | source_type={source_type}", flush=True)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, source_path, second, image_change, video_change, mode, source_type),
        daemon=True,
        name=f"pipeline-{job_id[:8]}",
    )
    thread.start()
    print(f"[{job_id[:8]}] Pipeline thread started: {thread.name}", flush=True)

    return jsonify({"job_id": job_id})


def _log(job_id: str, msg: str):
    print(f"[{job_id[:8]}] {msg}", flush=True)


def _set_step(job_id: str, index: int):
    jobs[job_id]["step"] = index + 1
    jobs[job_id]["step_name"] = STEPS[index]


def _run_pipeline(job_id: str, video_path: str, second: float, image_change, video_change, mode: str = "full", source_type: str = "video"):
    _log(job_id, f"=== Pipeline started | thread={threading.current_thread().name} | mode={mode} | source_type={source_type} ===")
    _log(job_id, f"Source: {video_path}")
    _log(job_id, f"Second: {second} | image_change: {image_change!r} | video_change: {video_change!r}")

    try:
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(exist_ok=True)

        if source_type == "image":
            # Use the uploaded image directly as the frame — skip Steps 1 and 2
            import shutil as _shutil
            frame_path = str(out_dir / "frame.jpg")
            _shutil.copy2(video_path, frame_path)
            jobs[job_id]["frame_ready"] = True
            _log(job_id, f"Image source: copied upload to {frame_path}")

            from PIL import Image as _PILImage
            with _PILImage.open(frame_path) as _img:
                _w, _h = _img.size
            aspect_ratio = _detect_ratio(_w, _h)
            _log(job_id, f"Image source: aspect_ratio={aspect_ratio} (from {_w}x{_h})")
        else:
            # Step 1 — extract frame
            _log(job_id, "Step 1: extracting frame from video...")
            _set_step(job_id, 0)
            frame_path = str(out_dir / "frame.jpg")
            extract_frame(video_path, second, frame_path)
            jobs[job_id]["frame_ready"] = True
            _log(job_id, f"Step 1 done: frame saved to {frame_path}")

            # Step 2 — aspect ratio
            _log(job_id, "Step 2: detecting aspect ratio...")
            _set_step(job_id, 1)
            aspect_ratio = get_aspect_ratio(video_path)
            _log(job_id, f"Step 2 done: aspect_ratio={aspect_ratio}")

        # Step 3 — GPT analysis
        _log(job_id, "Step 3: analyzing frame with GPT-5.5...")
        _set_step(job_id, 2)
        analysis = analyze_frame(frame_path, change=image_change, video_change=video_change)
        image_prompt = analysis.get("image_prompt", "")
        video_prompt = analysis.get("video_prompt", "")
        _log(job_id, f"Step 3 done: image_prompt length={len(image_prompt)} chars, video_prompt length={len(video_prompt)} chars")

        # ── Early exit: prompts only ───────────────────────────
        if mode == "prompts_only":
            _log(job_id, "Mode=prompts_only: stopping after GPT analysis")
            jobs[job_id].update({
                "status": "complete",
                "result": {
                    "image_prompt": image_prompt,
                    "video_prompt": video_prompt,
                    "aspect_ratio": aspect_ratio,
                },
            })
            _log(job_id, "=== Pipeline complete (prompts_only) ===")
            return

        # Step 4 — generate image
        _log(job_id, "Step 4: generating image via Magnific API...")
        _set_step(job_id, 3)
        generated_image, _ = generate_image_magnific(image_prompt, aspect_ratio, out_dir)
        if generated_image:
            jobs[job_id]["image_ready"] = True
            _log(job_id, f"Step 4 done: image saved to {generated_image}")
        else:
            _log(job_id, "Step 4: image generation skipped or failed (no MAGNIFIC_API_KEY or API error)")

        # ── Early exit: image only ─────────────────────────────
        if mode == "image_only":
            _log(job_id, "Mode=image_only: stopping after image generation")
            jobs[job_id].update({
                "status": "complete",
                "result": {
                    "image_prompt": image_prompt,
                    "video_prompt": video_prompt,
                    "aspect_ratio": aspect_ratio,
                },
            })
            _log(job_id, "=== Pipeline complete (image_only) ===")
            return

        # Step 5 — generate video (full mode only)
        _log(job_id, "Step 5: generating video via Kling 2.6 Pro...")
        _set_step(job_id, 4)
        generated_video = None
        if generated_image:
            generated_video = generate_video_magnific(generated_image, video_prompt, aspect_ratio, out_dir)
            if generated_video:
                jobs[job_id]["video_ready"] = True
                _log(job_id, f"Step 5 done: video saved to {generated_video}")
            else:
                _log(job_id, "Step 5: video generation failed or timed out")
        else:
            _log(job_id, "Step 5: skipped (no generated image)")

        # Step 6 — complete
        _set_step(job_id, 5)
        jobs[job_id].update({
            "status": "complete",
            "result": {
                "image_prompt": image_prompt,
                "video_prompt": video_prompt,
                "aspect_ratio": aspect_ratio,
            },
        })
        _log(job_id, "=== Pipeline complete (full mode) ===")

    except SystemExit as exc:
        error_msg = (
            "Pipeline exited unexpectedly. "
            "Ensure ffmpeg is installed and the video file is valid."
        )
        _log(job_id, f"ERROR (SystemExit code={exc.code}): {error_msg}")
        jobs[job_id].update({"status": "error", "error": error_msg})

    except Exception as exc:
        tb = traceback.format_exc()
        _log(job_id, f"ERROR ({type(exc).__name__}): {exc}")
        _log(job_id, f"Traceback:\n{tb}")
        jobs[job_id].update({"status": "error", "error": str(exc)})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/result/<job_id>")
def result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "complete":
        return jsonify({"error": "Job not complete", "status": job["status"]}), 202
    return jsonify(job["result"])


@app.route("/result/<job_id>/frame")
def result_frame(job_id):
    path = OUTPUT_DIR / job_id / "frame.jpg"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/result/<job_id>/image")
def result_image(job_id):
    path = OUTPUT_DIR / job_id / "generated_image.jpg"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/result/<job_id>/video")
def result_video(job_id):
    path = OUTPUT_DIR / job_id / "generated_video.mp4"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="video/mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
            request_handler=LongTimeoutHandler)
