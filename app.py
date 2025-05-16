import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf


# To run : python -m streamlit run app.py --logger.level error

# Load model and labels
model = tf.keras.models.load_model("model.h5", compile=False, safe_mode=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("Real-time Facial Emotion Detection")
st.markdown("Using webcam + Deep Learning + Streamlit WebRTC.")


class EmotionDetector(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.reshape(1, 48, 48, 1) / 255.0
            prediction = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            

            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            print("Current Emotion: ",emotion)
            break  # Only show first face

        if len(faces) == 0:
            st.session_state.predicted_emotion = "No face detected"

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="emotion-detect",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)
