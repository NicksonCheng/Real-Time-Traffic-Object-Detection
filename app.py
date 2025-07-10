from flask import Flask, Response, render_template
from ultralytics import YOLO
import threading
import cv2
import subprocess
import numpy as np
import time

app = Flask(__name__)

videos = [
    "https://hls.bote.gov.taipei/stream/004/index.m3u8",
    "https://hls.bote.gov.taipei/stream/004/index.m3u8",
]

model = YOLO("yolov8n.pt")
frame_buffers = [None for _ in videos]  # 儲存即時畫面
frame_locks = [threading.Lock() for _ in videos]  # 避免 race condition

# 控制推論頻率（秒）
INFERENCE_INTERVAL = 0.2  # 每張 frame 間隔至少 0.2 秒（約 5 FPS）


def video_worker(index, url):
    headers = "User-Agent: Mozilla/5.0\r\n" "Referer: https://bote.gov.taipei/"

    while True:
        try:
            ffmpeg_cmd = [
                "ffmpeg",
                "-headers",
                "Referer: https://hls.bote.gov.taipei/live/index.html?id=004\r\nUser-Agent: Mozilla/5.0",
                "-i",
                url,
                "-loglevel",
                "quiet",
                "-an",
                "-f",
                "image2pipe",
                "-pix_fmt",
                "bgr24",
                "-vcodec",
                "rawvideo",
                "-",
            ]

            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

            width, height = 720, 480
            frame_size = width * height * 3

            last_inference_time = 0

            while True:
                raw_frame = pipe.stdout.read(frame_size)
                if not raw_frame:
                    raise RuntimeError("FFmpeg pipe broken or empty.")

                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (height, width, 3)
                )

                now = time.time()
                if now - last_inference_time >= INFERENCE_INTERVAL:
                    results = model(frame)
                    annotated = results[0].plot()

                    with frame_locks[index]:
                        frame_buffers[index] = frame

                    last_inference_time = now

        except Exception as e:
            print(f"[警告] 串流 {index} 發生錯誤：{e}，3 秒後重啟...")
            time.sleep(3)


def generate(index):
    while True:
        with frame_locks[index]:
            frame = frame_buffers[index]

        if frame is None:
            time.sleep(0.1)
            continue

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        time.sleep(0.05)  # 控制前端畫面更新速度


@app.route("/")
def index():
    return render_template("index.html", num_videos=len(videos))


@app.route("/video_feed/<int:video_id>")
def video_feed(video_id):
    return Response(
        generate(video_id), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# 避免 Chrome 開發者工具錯誤
@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def ignore_devtools():
    return "", 204


if __name__ == "__main__":
    for idx, url in enumerate(videos):
        threading.Thread(target=video_worker, args=(idx, url), daemon=True).start()

    app.run(host="0.0.0.0", port=5003, threaded=True)
