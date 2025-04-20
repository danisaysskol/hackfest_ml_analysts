from ultralytics import YOLO

# Load the best trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Validate on validation set
metrics_1 = model.val()  # This uses the val split from data.yaml

# Print metrics
print("Validation Metrics:", metrics_1)
