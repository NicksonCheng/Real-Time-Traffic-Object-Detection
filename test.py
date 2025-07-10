import subprocess
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

m3u8_url = "https://hls.bote.gov.taipei/stream/004/index.m3u8"

ffmpeg_cmd = [
    "ffmpeg",
    "-headers",
    "Referer: https://hls.bote.gov.taipei/live/index.html?id=004\r\nUser-Agent: Mozilla/5.0",
    "-i",
    m3u8_url,
    "-f",
    "image2pipe",
    "-pix_fmt",
    "bgr24",
    "-vcodec",
    "rawvideo",
    "-",
]

# 解析度要跟來源一致，否則抓 frame 會錯
width, height = 720, 480
frame_size = width * height * 3

pipe = subprocess.Popen(
    ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
)
frame_id = 0
while True:
    raw_frame = pipe.stdout.read(frame_size)
    if not raw_frame:
        break
    frame_id += 1
    if frame_id % 5 != 0:
        continue  # 每5幀跑1次 YOLO
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

    results = model(frame)[0]
    annotated = results.plot()

    cv2.imshow("Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipe.terminate()
cv2.destroyAllWindows()
