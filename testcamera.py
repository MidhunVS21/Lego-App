import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Define the WebRTC streamer
webrtc_streamer(
    key="camera",  # A unique key for this streamer instance
    mode=WebRtcMode.SENDRECV,  # Send and receive mode to capture the camera feed
    media_stream_constraints={
        "video": True,  # Enable video feed
        "audio": False  # Disable audio feed
    }
)

st.write("Live Camera Feed")
