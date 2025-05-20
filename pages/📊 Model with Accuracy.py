import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import altair as alt
import time

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model4.keras", compile=False)

model = load_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# UI Title
st.title("Real-time Facial Emotion Detection with Bar Chart")

# Define the video processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_emotion = "No face detected"
        self.prediction = np.zeros(7)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            self.latest_emotion = "No face detected"
            self.prediction = np.zeros(7)
        else:
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_input = face_roi.reshape(1, 48, 48, 1) / 255.0

                predictions = model.predict(face_input, verbose=0)
                emotion = emotion_labels[np.argmax(predictions)]
                self.latest_emotion = emotion
                self.prediction = predictions[0]
                
                # Draw rectangle and label
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Only first face

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Create processor instance
processor_instance = EmotionProcessor()

# Layout with columns
col1, col2 = st.columns(2)

# Webcam feed with increased video size
with col1:
    ctx = webrtc_streamer(
        key="emotion",
        video_processor_factory=lambda: processor_instance,
        media_stream_constraints={"video": True, "audio": False},
        
    )

# Emotion chart display
with col2:
    st.subheader("📈 Accuracy Percentage")
    chart_placeholder = st.empty()

if ctx:
    while ctx.state.playing:
        prediction = ctx.video_processor.prediction

        if isinstance(prediction, (list, np.ndarray)) and len(prediction) == 7:
            df = pd.DataFrame({
                "Emotion": emotion_labels,
                "Probability": prediction
            })
            df['Percentage'] = df['Probability'].apply(lambda x: f"{x*100:.1f}%")

            bar_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Emotion', sort=emotion_labels),
                y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('Emotion', legend=None)
            )

            text = alt.Chart(df).mark_text(
                align='center',
                baseline='bottom',
                dy=-8,  # Move text slightly above the bar
                fontSize=12,
                fontWeight='bold',
                color='white'
            ).encode(
                x='Emotion',
                y='Probability',
                text='Percentage'
            )

            chart = (bar_chart + text).properties(width=400, height=300)
            chart_placeholder.altair_chart(chart, use_container_width=True)

        time.sleep(0.5)
