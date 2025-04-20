from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Evaluate on test set
metrics_2 = model.val(split='test')  # Uses test split from data.yaml

# Print test set metrics
print("Test Set Metrics:", metrics_2)
