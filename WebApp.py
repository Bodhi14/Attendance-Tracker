import streamlit as st

from training import train
from test import test


st.title("Face Recognition Attendance System")

# Training Section
st.header("Training")
name = st.text_input("Enter Your Name: ")
num_images = st.number_input("Number of Images to Capture:", min_value=1, step=10, value=100)
if st.button("Start Training"):
    train(name, num_images)
    st.success("Training Completed!")

# Recognition Section
st.header("Recognition")
if st.button("Start Recognition"):
    test()
    st.success("Recognition Completed!")