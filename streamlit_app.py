import streamlit as st
from PIL import Image
import numpy as np
import torch
from pathlib import Path

from src.FAZSegmentator import FAZSegmentator

st.set_page_config(layout='wide')

# Donlaod model
cloud_model_location = "1SxJQeT37bgdJgvZHAqrhNnPRBbaqT_Ax"

@st.cache
def load_model():

    save_dest = Path('models')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("models/Se_resnext50-920eef84.pth")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from src.GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)

load_model()

# LOAD MODEL
MODELPATH = 'models/Se_resnext50-920eef84.pth'
segmentator = FAZSegmentator(model_path=MODELPATH)

### ---------- RENDER ----------####
# SIDE BAR
st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

# MAIN COMPONENT
## TOP
st.write('# FAZ Segmentation Tool')

## BODY
st.markdown('##')
if uploaded_file is None:
    # Default image.
    image = np.ones((360,640))
    st.image(image, use_column_width=True)    
else:
	# User-selected image.
	image = Image.open(uploaded_file).convert("RGB")
	image = np.array(image)

	left_im, right_im = st.beta_columns(2)
	left_im.write('### Uploaded image')
	left_im.image(image, use_column_width=True)
	enhanced_image, phi, new_phi = segmentator.predict(image)
	right_im.write('### Enhanced image')
	right_im.image(enhanced_image, use_column_width=True)

	left_seg, right_seg = st.beta_columns(2)
	left_seg.write('### UNet')
	left_seg.image(phi, use_column_width=True)
	right_seg.write('### UNet + LevelSet')
	right_seg.image(new_phi, use_column_width=True)
