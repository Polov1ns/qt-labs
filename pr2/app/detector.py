import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(image_path: str, conf_threshold: float = 0.25):
    start_time = time.time()
    
    results = model(image_path, conf=conf_threshold)
    
    inference_time = time.time() - start_time
    detected_objects = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detected_objects.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "box": [round(coord, 2) for coord in xyxy]
            })
            
    return {
        "objects": detected_objects,
        "count": len(detected_objects),
        "inference_time_sec": round(inference_time, 3)
    }