import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os


MODEL_PATH = r"E:\submission\Part_1_Glove_Detection\model\glove_detection\weights\best.pt"
model = YOLO(MODEL_PATH)

st.title("Glove Detection App")
st.write("Upload an image to detect gloved and bare hands.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # ✅ updated
    
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
 
    results = model.predict(img, conf=0.5, verbose=False)
    
    glove_count = 0
    bare_count = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label.lower() in ["hand-in-glove", "gloved_hand"]:
                glove_count += 1
            elif label.lower() in ["hand", "bare_hand"]:
                bare_count += 1
    

    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img, caption='Annotated Image', use_container_width=True)  # ✅ updated

    st.write(f"**Gloved hands:** {glove_count}")
    st.write(f"**Bare hands:** {bare_count}")
