from urllib.parse import _NetlocResultMixinStr
from xml.etree.ElementTree import TreeBuilder
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import time
import io
import tempfile
import random
import base64
from PIL import Image
from tqdm import tqdm
from functions2 import *
import pickle
import update
import train
import predict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Import saved (pre-trained) model
path = ["best_model.16-0.43.h5", '-']


image = Image.open('logo.png')
st.set_page_config(page_title = "Dashboard", page_icon = image, layout = "wide")
 
# Set standard frame width and height
frame_width = 600
frame_height = 175

# Setting the font size of markdown
st.markdown("""
<style>
.medium-font {
    font-size:20px !important;
}
.small-font {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

#User interface
col1, col2, col3 = st.columns([3,6,2])

with col1: 
    st.write("")

with col2:
    st.image('new_logo.png', width = 800)

with col3:
    st.write("")

col1, col2, col3 = st.columns([2,10,1])
with col3:
    st.write("")
with col1:
    st.write("")
with col2:
    st.markdown('<p class="medium-font">Platform for general use, both with medical related and general engineering data sources to aid the analysis of data through the use of Neural Networks.', unsafe_allow_html= True)

# Download user manual 
st.sidebar.header("User Manual")
with open("User Manual.pdf", "rb") as file:
    st.sidebar.download_button("Download here", file, "User Manual.pdf")

# st.sidebar.header("What to do?")
# task = st.sidebar.selectbox("Select task:", ["-", "Train model", "Update model", "Make prediction"])

PAGES = {
    "Train model": train,
    "Update model": update,
    "Make predictions": predict
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
