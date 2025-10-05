# Glove Detection Project

### Dataset Name and Source
**Dataset:** Hand-Handinglove Detection Dataset  
**Source:** [Roboflow Universe](https://universe.roboflow.com/detr-cjz4w/hand-handinglove-detection)  

The dataset contains images of gloved and bare hands, split into train, validation, and test folders.

### Model Used
**Model:** YOLOv8 (pretrained `yolov8n.pt`)  
**Task:** Object detection to classify hands as `gloved_hand` or `bare_hand`.

### Preprocessing / Training Done
- Dataset organized in YOLO format (images + labels).  
- Trained on custom dataset for 2 epochs.  
- Confidence threshold set to 0.5 for detection.  
- Data augmentations were applied automatically by YOLOv8 during training.

### What Worked and What Didn’t
✅ **Worked:**  
- Detecting gloved vs bare hands accurately.  
- Annotated images and JSON logs generated successfully.  
- Streamlit UI works interactively for single image testing.  

❌ **Didn’t Work / Challenges:**  
- Training on CPU is slow (GPU recommended).  
- Some very small or blurry hands are occasionally missed.

### How to Run Your Script

#### 1️⃣ Activate Environment

cd E:\submission
python -m venv glove
.\glove\Scripts\activate
pip install -r requirements.txt

#### 2️⃣ Run Detection Script

Use this command to perform glove detection on test images:
python Part_1_Glove_Detection\detection_script.py \
--input "dataset\test\images" \
--output "output" \
--logs "logs" \
--model "model\glove_detection\weights\best.pt" \
--confidence 0.5

### 3️⃣ Run Streamlit App
streamlit run Part_1_Glove_Detection\app.py

