import streamlit as st
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import copy

from src.FAZSegmentator import FAZSegmentator
from src.utils import CustomBinarize

CONVERTION_FACTOR = 0.0055

#
st.set_page_config(layout='wide')

# ----------- Dowload and load model ------------------
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

	MODELPATH = 'models/Se_resnext50-920eef84.pth'
	segmentator = FAZSegmentator(model_path=MODELPATH)
	return segmentator

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

def analyze(image, method='UNet + LevelSet', segmentator=None):
	image_np = np.array(image)
	H, W, _ = image_np.shape
	area = H*W*convertion_factor**2

	# Original image and enhanced one
	cols = st.beta_columns(4)
	enhanced_image, mask, phi_erosed = segmentator.predict(image_np) # processing
	cols[0].write('### Uploaded image')
	cols[0].image(image)
	cols[1].write('### Enhanced image')
	cols[1].image(enhanced_image)
	cols[2].write('### Info')
	with cols[2]:
		st.write('Image dimention (pixels): \
				{}(H) x {}(W)'.format(H, W))
		st.write('Area: ', round(area, 3), ' $mm^2$')

	if method == 'UNet + LevelSet':
		processed = phi_erosed
	else:
		processed = mask

	# Unet + Levelset
	st.write(f"### {method} Segmentation")
	cols = st.beta_columns(4)
	cols[0].image(processed)
	cols[1].image(contourize(enhanced_image, processed))
	FAZpixel = np.sum(np.array(processed)/255)
	FAZarea = FAZpixel*(convertion_factor**2)
	cols[3].write('### Analysis Results')
	with cols[3]:
		st.write('FAZ Area: ', round(FAZarea,3), ' $mm^2$')
	
	# Calc Vessel Area Density (VAD)
	BW = CustomBinarize(np.array(enhanced_image.convert('L')))
	cols[2].image(BW)
	vesselpixel = np.count_nonzero(BW==255)
	VAD = vesselpixel/(H*W)*100
	with cols[3]:
		st.write('Vessel Area Density: ', round(VAD,2), '%')
	#del segmentator

### ---------- RENDER ----------####
segmentator = copy.deepcopy(load_model())

# SIDE BAR
st.sidebar.write('#### Upload and image')
uploaded_file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
sample = st.sidebar.selectbox(
	'or Select a sample',
	('-', '1.png', '2.png', '3.png'))
convertion_factor = st.sidebar.text_input('Conversion factor', value=str(CONVERTION_FACTOR))
convertion_factor = float(convertion_factor)

method = st.sidebar.radio("Choose Method",
		('UNet + LevelSet', 'UNet'))

# MAIN COMPONENT
## TOP
st.write('# FAZ Segmentation Tool')

USE_COLUMN_WIDTH = True

## BODY
st.markdown('##')

# Select a sample
if sample is not '-':
	impath = 'samples/' + sample
	image = Image.open(impath).convert("RGB")
	analyze(image, method, segmentator)

# Upload a file
if uploaded_file is None:
    # Default image.
    image = np.ones((360,640))
    st.image(image, use_column_width=USE_COLUMN_WIDTH)    
else:
	# User-selected image.
	image = Image.open(uploaded_file).convert("RGB")
	analyze(image, method, segmentator)
