import streamlit as st
from roboflow import Roboflow
import pandas as pd
from pathlib import Path
import os

# Set up Roboflow
rf = Roboflow(api_key="nB67mc0eYz8FHOWIS0A7")
project = rf.workspace().project("construction-ppe-rdhzo")
model = project.version(1).model

def predict_file(file_path, confidence=40, overlap=30):
    file_type = Path(file_path).suffix.lower()

    if file_type in ['.jpg', '.jpeg', '.png']:
        result = model.predict(file_path, confidence=confidence, overlap=overlap).json()
    elif file_type in ['.mp4', '.avi', '.mov']:
        # Implement video prediction logic here if needed
        result = model.predict_video(file_path)
    else:
        print("Unsupported file type")

    return result

def save_uploaded_file(uploaded_file):
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("Helmet Detection App")
    file1 = st.file_uploader("Upload an Image or Video File")

    if file1:
        file_path = save_uploaded_file(file1)
        image_result = predict_file(file_path)

        class_list = [prediction['class'] for prediction in image_result['predictions']]
        confidence_list = [prediction['confidence'] for prediction in image_result['predictions']]

        res = pd.DataFrame(list(zip(class_list, confidence_list)), columns=['cat', 'conf'])
        st.write(res)

        for i in range(len(res)):
            if res['cat'][i] == "no hat" and res['conf'][i] >= 0.5:
                st.warning("No Helmet Detected! ⚠️")

if __name__ == "__main__":
    main()
