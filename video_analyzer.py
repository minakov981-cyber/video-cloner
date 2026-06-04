import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from math import gcd
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAGNIFIC_API_KEY = os.getenv("MAGNIFIC_API_KEY")
MAGNIFIC_BASE = "https://api.magnific.com/v1/ai/text-to-image/nano-banana-pro"
MAGNIFIC_VIDEO_POST = "https://api.magnific.com/v1/ai/image-to-video/kling-v2-6-pro"
MAGNIFIC_VIDEO_STATUS_BASE = "https://api.magnific.com/v1/ai/image-to-video/kling-v2-6"


def normalize_aspect_ratio(ratio: str) -> str:
    if ratio == "16:9":
        return "widescreen_16_9"
    if ratio == "9:16":
        return "social_story_9_16"
    try:
        w, h = map(int, ratio.split(":"))
        if w > h:
            return "widescreen_16_9"
        if h > w:
            return "social_story_9_16"
    except Exception:
        pass
    return "square_1_1"


def extract_frame(video_path: str, second: int, output_path: str) -> str:
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install it with: brew install ffmpeg")
        sys.exit(1)

    cmd = [
        "ffmpeg",
        "-ss", str(second),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
        "-y",
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        sys.exit(1)

    if not os.path.isfile(output_path):
        print(f"Error: frame not extracted. Check that second {second} is within video duration.")
        sys.exit(1)

    return output_path


def _detect_ratio(w: int, h: int) -> str:
    if not w or not h:
        return "1:1"
    ratio = w / h
    candidates = {
        "1:1":  1.0,
        "16:9": 16/9,
        "9:16": 9/16,
        "4:3":  4/3,
        "3:4":  3/4,
        "3:2":  3/2,
        "2:3":  2/3,
        "4:5":  4/5,
        "5:4":  5/4,
        "21:9": 21/9,
    }
    return min(candidates, key=lambda k: abs(candidates[k] - ratio))


def get_aspect_ratio(video_path: str) -> str:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return "1:1"

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return "1:1"

    w = streams[0].get("width", 0)
    h = streams[0].get("height", 0)
    return _detect_ratio(w, h)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frame(frame_path: str, change: str = None, video_change: str = None) -> dict:
    print("Analyzing frame with GPT-5.5...")
    b64 = encode_image(frame_path)

    change_instruction = (
        f"\n\nIMPORTANT CHANGE for image_prompt ONLY: Apply this modification when writing image_prompt: {change}\n"
        "Keep all other scene details exactly as seen in the image. Only replace or modify the specified element."
    ) if change else ""

    video_change_instruction = (
        f"\n\nIMPORTANT CHANGE for video_prompt ONLY: Apply this modification when writing video_prompt: {video_change}\n"
        "Keep all other scene details exactly as seen in the image. Only replace or modify the specified element."
    ) if video_change else ""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional visual analyst and AI prompt engineer. "
                    "IMPORTANT: Completely ignore any text, titles, subtitles, captions, watermarks, logos, or overlays visible on the image. "
                    "Analyze ONLY the underlying visual scene: subjects, environment, lighting, colors, composition, camera angle, and atmosphere. "
                    "Treat the image as if no text or overlays exist."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and return ONLY valid JSON (no markdown, no code fences):\n"
                            "{\n"
                            '  "image_prompt": "A highly detailed English prompt for an image generator (Midjourney/DALL-E/Flux). '
                            "Describe: exact composition, visual style, genre, mood, color palette, color grading, lighting setup and direction, "
                            "camera angle and lens, exact number of people and objects visible, their positions, clothing, expressions, "
                            'background/environment details, time of day, atmosphere. 150-250 words.",\n'
                            '  "video_prompt": "A highly detailed English prompt optimized for Kling 2.6/3.0 on Loveart. '
                            "Based on this frame, describe the full scene to recreate or extend as a video clip. Include: "
                            "subjects and their actions, environment, visual style, color grading, lighting, camera movement (pan/tilt/zoom/dolly/static/handheld), "
                            'motion intensity and dynamics, mood and atmosphere. 150-250 words."\n'
                            f"}}{change_instruction}{video_change_instruction}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())


def generate_image_magnific(prompt: str, aspect_ratio: str, out_dir: Path) -> tuple:
    if not MAGNIFIC_API_KEY:
        print("Warning: MAGNIFIC_API_KEY not set — skipping image generation.")
        return None, None

    headers = {"x-magnific-api-key": MAGNIFIC_API_KEY}

    print("Sending request to Magnific API...")
    response = requests.post(
        MAGNIFIC_BASE,
        headers=headers,
        json={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": "2K",
        },
    )
    if not response.ok:
        print(f"Magnific API error {response.status_code}: {response.text}")
        return None, None
    task_id = response.json()["data"]["task_id"]
    print(f"Magnific task_id: {task_id}")

    print("Waiting for generation to complete", end="", flush=True)
    while True:
        time.sleep(5)
        status_response = requests.get(
            f"{MAGNIFIC_BASE}/{task_id}",
            headers=headers,
        )
        status_response.raise_for_status()
        data = status_response.json().get("data", status_response.json())
        status = data.get("status", "")
        print(".", end="", flush=True)

        if status == "COMPLETED":
            print(" done.")
            generated = data.get("generated", [])
            image_url = generated[0] if generated else data.get("image_url", "")
            break
        elif status in ("FAILED", "ERROR"):
            print(f" failed.\nMagnific error: {data}")
            return None, None

    image_path = str(out_dir / "generated_image.jpg")
    img_data = requests.get(image_url).content
    with open(image_path, "wb") as f:
        f.write(img_data)
    print(f"Generated image saved: {image_path}")
    return image_path, image_url


def generate_video_magnific(image_path: str, prompt: str, aspect_ratio: str, out_dir: Path) -> str:
    if not MAGNIFIC_API_KEY:
        print("Warning: MAGNIFIC_API_KEY not set — skipping video generation.")
        return None

    headers = {"x-magnific-api-key": MAGNIFIC_API_KEY}
    kling_aspect = normalize_aspect_ratio(aspect_ratio)

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    print("Sending request to Kling 2.6 Pro (image-to-video)...")
    response = requests.post(
        MAGNIFIC_VIDEO_POST,
        headers=headers,
        json={
            "prompt": prompt,
            "image": b64_image,
            "duration": "5",
            "aspect_ratio": kling_aspect,
            "generate_audio": False,
        },
    )
    if not response.ok:
        print(f"Kling API error {response.status_code}: {response.text}")
        return None
    task_id = response.json()["data"]["task_id"]
    print(f"Kling task_id: {task_id}")

    max_wait_seconds = 900
    elapsed = 0
    poll_interval = 10
    print("Waiting for video generation to complete", end="", flush=True)
    video_url = None
    while elapsed < max_wait_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval
        status_response = requests.get(
            f"{MAGNIFIC_VIDEO_STATUS_BASE}/{task_id}",
            headers=headers,
        )
        if status_response.status_code == 404:
            print(".", end="", flush=True)
            continue
        status_response.raise_for_status()
        data = status_response.json().get("data", status_response.json())
        status = data.get("status", "")
        print(".", end="", flush=True)

        if status == "COMPLETED":
            print(" done.")
            generated = data.get("generated", [])
            video_url = generated[0] if generated else ""
            if not video_url:
                print("Error: no video URL in response.")
                return None
            break
        elif status in ("FAILED", "ERROR"):
            print(f" failed.\nKling error: {data}")
            return None

    if not video_url:
        print(f"\nTimeout: video not ready after {max_wait_seconds}s.")
        task_file = str(out_dir / "video_task_id.txt")
        with open(task_file, "w") as f:
            f.write(f"task_id: {task_id}\nstatus_url: {MAGNIFIC_VIDEO_STATUS_BASE}/{task_id}\n")
        print(f"Task ID saved to: {task_file}")
        return None

    video_path = str(out_dir / "generated_video.mp4")
    video_data = requests.get(video_url).content
    with open(video_path, "wb") as f:
        f.write(video_data)
    print(f"Generated video saved: {video_path}")
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Extract a frame and generate image/video prompts via GPT-5.5")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--second", type=int, default=0, help="Second to extract the frame from (default: 0)")
    parser.add_argument("--change", type=str, default=None, help="Modification to apply to image_prompt only, e.g. 'replace woman with a man in a suit'")
    parser.add_argument("--video-change", type=str, default=None, help="Modification to apply to video_prompt only, e.g. 'add slow motion camera movement'")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    video_stem = Path(args.video).stem
    out_dir = Path(f"{video_stem}_s{args.second}")
    out_dir.mkdir(exist_ok=True)
    print(f"Output folder: {out_dir}/")

    print(f"Video: {args.video}")
    aspect_ratio = get_aspect_ratio(args.video)
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Extracting frame at second {args.second}...")

    frame_path = str(out_dir / "frame.jpg")
    extract_frame(args.video, args.second, frame_path)
    print(f"Frame saved: {frame_path}")

    if args.change:
        print(f"Image change: {args.change}")
    if args.video_change:
        print(f"Video change: {args.video_change}")

    result = analyze_frame(frame_path, change=args.change, video_change=args.video_change)

    image_prompt = result.get("image_prompt", "")
    video_prompt = result.get("video_prompt", "")

    with open(out_dir / "image_prompt.txt", "w", encoding="utf-8") as f:
        f.write(image_prompt)

    with open(out_dir / "video_prompt.txt", "w", encoding="utf-8") as f:
        f.write(video_prompt)

    print("\n--- IMAGE PROMPT ---")
    print(image_prompt)
    print("\n--- VIDEO PROMPT (Kling 2.6/3.0) ---")
    print(video_prompt)

    generated_image, generated_image_url = generate_image_magnific(image_prompt, aspect_ratio, out_dir)

    generated_video = None
    if generated_image:
        generated_video = generate_video_magnific(generated_image, video_prompt, aspect_ratio, out_dir)

    saved = ["frame.jpg", "image_prompt.txt", "video_prompt.txt"]
    if generated_image:
        saved.append("generated_image.jpg")
    if generated_video:
        saved.append("generated_video.mp4")
    print(f"\nSaved to: {out_dir}/ → {', '.join(saved)}")


if __name__ == "__main__":
    main()
