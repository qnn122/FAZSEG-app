from numpy.core.fromnumeric import resize
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.efficientunet import *
from src.model import get_torchvision_model

# FAZ Preprocessing
from src.dataset import FAZ_Preprocess

# Level Set
from src.levelset import levelset_optmized


class FAZSegmentator():
	'''
	FAZ Segmentation 
	'''
	def __init__(self, model_type='Se_resnext50', model_path=None, using_gpu=True):
		'''
		Load trained model
		'''
		# Check device
		self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'
		# Create model
		
		self.model = get_torchvision_model(model_type, True, 1, "focal")

		# Convert to DataParallel and move to CPU/GPU
		#self.model = nn.DataParallel(self.model).to(self.device)

		# Load trained model
		if model_path is not None:
			state_dict = torch.load(model_path, map_location=self.device)
			state_dict = state_dict["state"]
			self.model.load_state_dict(state_dict)

		# Switch model to evaluation mode
		self.model.eval()

		# Image processing
		self.size = 256
		self.transform = transforms.Compose([
			transforms.Resize((self.size, self.size)),
			transforms.ToTensor()
		])

	def resize(self, im):
		'''
		im: numpy array
		H, W: original H and W
		'''
		im = Image.fromarray(im.astype(np.float32)*255).convert("RGB")
		im_resized = im.resize((self.H, self.W), resample=0)
		return im_resized

	def predict(self, image_input):
		'''
		Predict image in image_path is left or right
		image_input can be numpy array or image path
		'''
		self.H, self.W, self.c = image_input.shape # original shape

		# Load image
		raw_image = FAZ_Preprocess(image_input,[0.5,1, 1.5, 2, 2.5],1, 2)
		raw_image = raw_image.vesselness2d()
		enhanced_image = Image.fromarray(raw_image.astype(np.float32)*255).convert("RGB")
		enhanced_image = self.transform(enhanced_image)

		# Prediction model:
		mask = enhanced_image.unsqueeze(0)
		with torch.no_grad():
			mask = self.model(mask)
		mask = (mask.to(self.device).detach().numpy() > 0.6)*1
		mask = mask.reshape((self.size, self.size))

		# Prepare image for visualization
		enhanced_image = enhanced_image.permute(1,2,0).numpy()
		enhanced_image =  cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
		levelset_img = enhanced_image.copy()
		_, phi_erosed = levelset_optmized(levelset_img, mask)

		return self.resize(enhanced_image), self.resize(mask), self.resize(phi_erosed)
