from flask import Flask, Response, render_template
from ultralytics import YOLO
import threading
import cv2
import time

app = Flask(__name__)

# List of video sources
videos = [
    "https://hls.bote.gov.taipei/hls/036/index.m3u8",
    "https://hls.bote.gov.taipei/hls/031/index.m3u8",
    # Add more video URLs
]

model = YOLO("runs/detect/train/weights/best.pt")
frame_buffers = [None for _ in videos]  # Shared frame per stream


def video_worker(index, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[{index}] Failed to open stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{index}] Frame read failed, retrying...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(url)
            continue

        # Optional: Skip frames for speed
        # if frame_counter % 5 != 0: continue

        results = model(frame)
        annotated = results[0].plot()
        frame_buffers[index] = annotated


def generate(index):
    while True:
        frame = frame_buffers[index]
        if frame is None:
            time.sleep(0.1)
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.05)  # ~20 FPS


@app.route("/")
def index():
    return render_template("index.html", num_videos=len(videos))


@app.route("/video_feed/<int:video_id>")
def video_feed(video_id):
    return Response(
        generate(video_id), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # Start threads for each video stream
    for idx, url in enumerate(videos):
        threading.Thread(target=video_worker, args=(idx, url), daemon=True).start()

    app.run(host="0.0.0.0", port=5000, threaded=True)
