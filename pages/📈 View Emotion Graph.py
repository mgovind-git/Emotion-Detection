import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import altair as alt


@st.cache_resource  # Use streamlit's resource caching
def load_model():
    return tf.keras.models.load_model("my_model4.keras", compile=False)

model = load_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Page config
st.set_page_config(layout="wide")
st.title("🎥 Real-Time Facial Emotion Detection with Live Graph")

# Initialize session state for emotion log
if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

# Sidebar controls
chart_type = st.sidebar.radio(
    "📊 Select Chart Type",
    ["Line Chart", "Bar Chart", "Pie Chart"],
    index=0
)

if st.sidebar.button("🧹 Clear Emotion History"):
    st.session_state.emotion_log = []
    st.sidebar.success("Emotion log cleared!")

# Prepare dataframe for download button (empty or with data)
def get_csv():
    if st.session_state.emotion_log:
        df_download = pd.DataFrame(st.session_state.emotion_log)
    else:
        df_download = pd.DataFrame(columns=["Time", "Emotion", "Confidence"])
    return df_download.to_csv(index=False).encode('utf-8')

csv = get_csv()

st.sidebar.download_button(
    label="⬇ Download Emotion Log (CSV)",
    data=csv,
    file_name="emotion_log.csv",
    mime="text/csv",
    disabled=not bool(st.session_state.emotion_log),  # disabled if no data
)

# Video processor class
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_emotion = "No face detected"
        self.latest_confidence = 0.0  # store confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            self.latest_emotion = "No face detected"
            self.latest_confidence = 0.0
        else:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.reshape(1, 48, 48, 1) / 255.0
                prediction = model.predict(face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = float(prediction[0][emotion_idx])  # highest softmax score

                self.latest_emotion = emotion
                self.latest_confidence = confidence

                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                # Show emotion and confidence percentage on video
                cv2.putText(image, f"{emotion} ({confidence*100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Instantiate processor
processor_instance = EmotionProcessor()

# Layout: webcam left, graph right
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📷 Live Webcam")
    ctx = webrtc_streamer(
        key="emotion-stream",
        video_processor_factory=lambda: processor_instance,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📈 Emotion Graph")
    emotion_display = st.empty()
    chart_placeholder = st.markdown("No data to display yet!")

# Real-time loop for updating emotion and graph
if ctx:
    while ctx.state.playing:
        if hasattr(ctx.video_processor, "latest_emotion") and hasattr(ctx.video_processor, "latest_confidence"):
            current_emotion = ctx.video_processor.latest_emotion
            current_confidence = ctx.video_processor.latest_confidence
            emotion_display.markdown(f"**Detected Emotion:** {current_emotion} ({current_confidence*100:.1f}%)")

            timestamp = time.strftime("%H:%M:%S")
            st.session_state.emotion_log.append({
                "Time": timestamp,
                "Emotion": current_emotion,
                "Confidence": round(current_confidence, 4)
            })

            df = pd.DataFrame(st.session_state.emotion_log[-50:])  # last 50 entries

            if chart_type == "Line Chart":
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X("Time", axis=alt.Axis(title="Time")),
                    y=alt.Y("Emotion", sort=emotion_labels, axis=alt.Axis(title="Emotion")),
                    color=alt.value("#008000")
                ).properties(height=300)

            elif chart_type == "Bar Chart":
                freq_df = df["Emotion"].value_counts().reset_index()
                freq_df.columns = ["Emotion", "Count"]
                chart = alt.Chart(freq_df).mark_bar().encode(
                    x="Emotion",
                    y="Count",
                    color=alt.Color("Emotion", scale=alt.Scale(scheme='category10'))
                ).properties(height=300)

            elif chart_type == "Pie Chart":
                freq_df = df["Emotion"].value_counts().reset_index()
                freq_df.columns = ["Emotion", "Count"]
                chart = alt.Chart(freq_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Emotion", type="nominal"),
                    tooltip=["Emotion", "Count"]
                ).properties(height=300)

            chart_placeholder.altair_chart(chart, use_container_width=True)
            time.sleep(0.5)
