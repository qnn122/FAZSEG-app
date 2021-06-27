import streamlit as st
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import copy

from src.FAZSegmentator import FAZSegmentator

st.set_page_config(layout='wide')

# ----------- Dowload model ------------------
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

# ----------- Some helpers -------------
def contourize(im, mask):
	'''
	fig, ax = plt.subplots()
	ax.imshow(im, cmap='gray')
	mask_2d = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY) 
	ax.contour(mask_2d, levels=np.logspace(-4.7, -3., 10), colors='white', alpha=0.2)
	ax.axis('off')
	ax.axis('tight')
	ax.axis("image")
	'''
	T = 0.5*256
	mask_np = np.array(mask)
	mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(mask_gray, T, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = np.array(contours[0]).reshape((-1,1,2)).astype(np.int32)
	imgout = cv2.drawContours(np.array(im), [cnt], -1, (0,255,0), 2)
	return imgout

### ---------- RENDER ----------####
# SIDE BAR
st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
converstion_factor = st.sidebar.text_input('Conversion factor', value='0.0055')


# MAIN COMPONENT
## TOP
st.write('# FAZ Segmentation Tool')

USE_COLUMN_WIDTH = True

## BODY
st.markdown('##')
if uploaded_file is None:
    # Default image.
    image = np.ones((360,640))
    st.image(image, use_column_width=USE_COLUMN_WIDTH)    
else:
	# User-selected image.
	image = Image.open(uploaded_file).convert("RGB")
	image_np = np.array(image)

	# Original image and enhanced one
	cols = st.beta_columns(4)
	enhanced_image, mask, phi_erosed = segmentator.predict(image_np) # processing
	cols[0].write('### Uploaded image')
	cols[0].image(image)
	cols[1].write('### Enhanced image')
	cols[1].image(enhanced_image)
	cols[2].write('### Info')
	cols[2].write('Image dimention: (123, 302)')
	cols[2].write('Size: 2.3043 m^2')
	cols[2].write('Area: 3938 m^2')

	# Unet + Levelset
	st.write("### UNet + LevelSet Segmentation")
	cols = st.beta_columns(4)
	cols[0].image(phi_erosed)
	cols[1].image(contourize(enhanced_image, phi_erosed))
	FAZpixel = 493
	FAZarea = 2.3948
	cols[2].image(phi_erosed)
	cols[3].write('### Analysis Results')
	cols[3].write('FAZ Area: **3.3948** mm^2')