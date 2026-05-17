import os
import uuid
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from video_analyzer import (
    extract_frame,
    get_aspect_ratio,
    analyze_frame,
    generate_image_magnific,
    generate_video_magnific,
)

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: dict = {}

STEPS = [
    "Extracting frame from video",
    "Detecting aspect ratio",
    "Analyzing frame with GPT-4o",
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
    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    try:
        second = float(request.form.get("second", 0))
    except (ValueError, TypeError):
        second = 0.0

    image_change = request.form.get("image_change") or None
    video_change = request.form.get("video_change") or None

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(exist_ok=True)

    filename = secure_filename(video_file.filename or "video.mp4")
    video_path = str(upload_path / filename)
    video_file.save(video_path)

    jobs[job_id] = {
        "status": "running",
        "step": 0,
        "step_name": "",
        "error": None,
        "frame_ready": False,
        "image_ready": False,
        "video_ready": False,
        "result": None,
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, video_path, second, image_change, video_change),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


def _set_step(job_id: str, index: int):
    jobs[job_id]["step"] = index + 1
    jobs[job_id]["step_name"] = STEPS[index]


def _run_pipeline(job_id: str, video_path: str, second: float, image_change, video_change):
    try:
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(exist_ok=True)

        # Step 1 — extract frame
        _set_step(job_id, 0)
        frame_path = str(out_dir / "frame.jpg")
        extract_frame(video_path, second, frame_path)
        jobs[job_id]["frame_ready"] = True

        # Step 2 — aspect ratio
        _set_step(job_id, 1)
        aspect_ratio = get_aspect_ratio(video_path)

        # Step 3 — GPT analysis
        _set_step(job_id, 2)
        analysis = analyze_frame(frame_path, change=image_change, video_change=video_change)
        image_prompt = analysis.get("image_prompt", "")
        video_prompt = analysis.get("video_prompt", "")

        # Step 4 — generate image
        _set_step(job_id, 3)
        generated_image, _ = generate_image_magnific(image_prompt, aspect_ratio, out_dir)
        if generated_image:
            jobs[job_id]["image_ready"] = True

        # Step 5 — generate video
        _set_step(job_id, 4)
        generated_video = None
        if generated_image:
            generated_video = generate_video_magnific(generated_image, video_prompt, aspect_ratio, out_dir)
            if generated_video:
                jobs[job_id]["video_ready"] = True

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

    except SystemExit:
        jobs[job_id].update({
            "status": "error",
            "error": "Pipeline exited unexpectedly. Ensure ffmpeg is installed and the video file is valid.",
        })
    except Exception as exc:
        jobs[job_id].update({
            "status": "error",
            "error": str(exc),
        })


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
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
