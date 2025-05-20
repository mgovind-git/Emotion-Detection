import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf

# To run: python -m streamlit run app.py --logger.level error
# streamlit run app.py --server.port 8502
# hide_streamlit_style = """
#     <style>
#     /* Hide main footer */
#     footer {visibility: hidden;}
#     /* Hide hamburger menu */
#     #MainMenu {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("my_model4.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load the model: {e}")
        st.stop()

model = load_model()


# Emotion labels and face detection
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    st.error(f"❌ Failed to load face cascade: {e}")
    st.stop()

# UI

st.title("Real-time Facial Emotion Detection")
st.markdown("Using webcam + Deep Learning + Streamlit WebRTC.")


class EmotionDetector(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) == 0:
                cv2.putText(image, "No face detected", (30, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    try:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (48, 48))
                        roi = roi_gray.reshape(1, 48, 48, 1).astype('float32') / 255.0
                        prediction = model.predict(roi, verbose=0)
                        emotion = emotion_labels[np.argmax(prediction)]
                        

                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(image, emotion, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        print("Current Emotion:", emotion)
                        print(f"Prediction: Angry - {prediction[0][0]:.2%} \nDisgust - {prediction[0][1]:.2%} \nFear - {prediction[0][2]:.2%} \nHappy - {prediction[0][3]:.2%} \nNeutral - {prediction[0][4]:.2%} \nSad - {prediction[0][5]:.2%} \nSurprise - {prediction[0][6]:.2%}")
                    except Exception as pred_err:
                        print(f"Prediction error: {pred_err}")
                        cv2.putText(image, "Prediction error", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    break  # Only process the first face
        except Exception as e:
            print(f"Frame processing error: {e}")
            cv2.putText(image, "Frame error", (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="emotion-detect",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)
