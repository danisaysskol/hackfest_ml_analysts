from ultralytics import YOLO

# Load YOLOv8 model (medium version)
model = YOLO('yolov8m.pt')

# Train the model with slightly tweaked hyperparameters
model.train(
    data="/content/hackfest-zxvsi/data.yaml",
    epochs=100,
    imgsz=640,
    batch=24,
    lr0=0.0012,    
    optimizer="AdamW",
    weight_decay=0.0008,     
    momentum=0.94,           
    dropout=0.08,            
    mosaic=0.25,             
    mixup=0.15,              
    hsv_h=0.02,              
    hsv_s=0.65,              
    hsv_v=0.45,         
    degrees=0.5,        
    translate=0.12,          
    scale=0.35,              
    shear=0.05,              
    flipud=0.0,
    fliplr=0.5,
    device=0,
    single_cls=False,
    freeze=0,
    val=True
)