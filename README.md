# FAZ-Segmentation
[**Bachground**](#background) | [**Setup environment**](#setup-environment) | [**Streamlit app**](#streamlit-app)

## Background
OCT Angiography (OCTA) has recently attracted a lot of attention as a new diagnostic method in ophthalmology. It is a non-invasive imaging technique that uses motion contrast to generate angiographic images in a matter of seconds without the need of contrast agents. OCTA thus has a great potential in diagnosing and monitoring various retinal diseases such as age-related macular degeneration, retinal vascular occlusions, diabetic retinopathy, glaucoma, and etc.

Fovea Avasclar Zone (FAZ) is a capillary free zone at the center of macula. Because of strong support of new imaging techniques, some authors have used vanilla computer vision algorithms to extract FAZ in OCTA image for vascular disease detections that affect the retinal microcirculation. Other researchers use deep learning to solve FAZ segmentation.

In this paper, we proposed a method which combines traditional computer vision techniques (Hessian Filter and Level Set) with a Unet-based model to automatically quantify the avascular zone in OCTA image. In specific, blood vessels in OCT based angiography images will be enhanced. Then we feed those images into a U-shape semantic segmentation model to extract the FAZ. 

A full report of this work can be found [here](https://drive.google.com/file/d/1owZMp2b_wBaOWNeudv_cxfOKU_Q8our-/view?usp=sharing)


## Setup Environment
Create a virtual environment and install dependency libraries
```bash
conda env create -f environment.yml
```

## Streamlit app
Run
```bash
streamlit run streamlit_app.py
```

The app is publicly avaiable at: https://share.streamlit.io/qnn122/FAZSEG-app/main/streamlit_app.py
