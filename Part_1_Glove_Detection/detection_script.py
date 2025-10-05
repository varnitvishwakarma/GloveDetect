import os
import json
import cv2
from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser(description="Glove Detection on Images")
parser.add_argument("--input", type=str, default=r"E:\submission\Part_1_Glove_Detection\dataset\test\images",
                    help="Path to folder with test images")
parser.add_argument("--output", type=str, default=r"E:\submission\Part_1_Glove_Detection\output",
                    help="Path to save annotated images")
parser.add_argument("--logs", type=str, default=r"E:\submission\Part_1_Glove_Detection\logs",
                    help="Path to save JSON logs")
parser.add_argument("--model", type=str, default=r"E:\submission\Part_1_Glove_Detection\model\glove_detection\weights\best.pt",
                    help="Path to trained YOLOv8 model (.pt)")
parser.add_argument("--confidence", type=float, default=0.5,
                    help="Minimum confidence threshold")
args = parser.parse_args()


os.makedirs(args.output, exist_ok=True)
os.makedirs(args.logs, exist_ok=True)


print(f"Loading YOLO model from {args.model} ...")
model = YOLO(args.model)

image_files = [f for f in os.listdir(args.input) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    img_path = os.path.join(args.input, img_file)
    print(f"\nProcessing: {img_path}")
    
  
    results = model.predict(img_path, conf=args.confidence, verbose=False)
    
    detections = []
    glove_count = 0
    bare_count = 0
    

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })

     
            if label.lower() in ["hand-in-glove", "gloved_hand"]:
                glove_count += 1
            elif label.lower() in ["hand", "bare_hand"]:
                bare_count += 1

    log_file = os.path.join(args.logs, img_file.rsplit(".", 1)[0] + ".json")
    with open(log_file, "w") as f:
        json.dump({
            "filename": img_file,
            "detections": detections,
            "gloved_hands": glove_count,
            "bare_hands": bare_count
        }, f, indent=4)
    

    annotated_img_path = os.path.join(args.output, img_file)
    annotated_img = results[0].plot()  
    cv2.imwrite(annotated_img_path, annotated_img)
    

    print(f"✅ Gloved hands: {glove_count}, Bare hands: {bare_count}")

print("\n✅ Detection completed. Annotated images saved to output/, logs saved to logs/")
