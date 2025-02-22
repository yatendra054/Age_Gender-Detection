import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
from PIL import Image,ImageOps
from keras.metrics import mean_absolute_error
import cv2



model = load_model('Age_Sex_detection1.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(frame):
    """Detect and extract the face from the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  
        face = frame[y:y+h, x:x+w]  
        return face, (x, y, w, h)
    
    return None, None 

def preprocess_image(image):
    image = image.convert('RGB')  
    image = image.resize((48, 48))  
    img_array = np.array(image)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    return img_array

def predict_age_gender(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    gender_labels = ['Male', 'Female']  
    age = int(np.round(predictions[1][0]))  
    sex = int(np.round(predictions[0][0]))  
    gender = gender_labels[sex]  
    return age, gender

st.title('Age and Gender Detection')
st.subheader('Upload an image for age and gender prediction')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image=ImageOps.exif_transpose(image)
    st.image(image, caption='Uploaded Image', use_column_width=False,width=250)
    age, gender = predict_age_gender(image)

    
    st.markdown(f"<h2 style='color:blue;'>Predicted Age: {age}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:green;'>Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
    
st.write("## WebCamera")

run = st.checkbox('Start WebCam')
FRAME_WINDOW = st.image([])

if run:
    camera = cv2.VideoCapture(0)
    captured=False

    while True:
        success, frame = camera.read()
        if not success:
            st.write("Failed to access camera.")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, coords = detect_face(frame)

        if face is not None and not captured:
            x, y, w, h = coords
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  
            pil_image = Image.fromarray(face)
            age, gender = predict_age_gender(pil_image)

            st.markdown(f"<h2 style='color:blue;'>Predicted Age: {age}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:green;'>Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
            
            
            FRAME_WINDOW.image(frame, caption="Detected Face", width=400)

            captured = True
            break 

        FRAME_WINDOW.image(frame, caption="Webcam Feed", width=400)

    camera.release()
    st.write("WebCam Stopped ✅")
    
    
