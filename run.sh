# Generate timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run training in background with logging
nohup yolo detect train model=yolov8n.pt data=datasets/traffic/traffic.yaml epochs=50 imgsz=640 > log/${timestamp}.txt 2>&1 &