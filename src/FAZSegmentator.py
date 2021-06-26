import torch
import PIL
import os
import cv2
from scipy import ndimage
import numpy as np
from PIL import Image
from torchvision import transforms
from src.efficientunet import *
from src.model import get_torchvision_model
from scipy.ndimage.morphology import binary_erosion

# FAZ Preprocessing
from src.dataset import FAZ_Preprocess

# Level Set
from src.levelset import *


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

	def predict(self, image_input):
		'''
		Predict image in image_path is left or right
		image_input can be numpy array or image path
		'''
		# Load image
		raw_image = FAZ_Preprocess(image_input,[0.5,1, 1.5, 2, 2.5],1, 2)
		raw_image = raw_image.vesselness2d()
		image = Image.fromarray(raw_image.astype(np.float32)*255).convert("RGB")
		image = self.transform(image)

		# Prediction model:
		mask = image.unsqueeze(0)
		with torch.no_grad():
			mask = self.model(mask)
		mask = (mask.to(self.device).detach().numpy() > 0.6)*1
		mask = mask.reshape((self.size,self.size))

		# Prepare image for visualization
		image = image.permute(1,2,0).numpy()
		image =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		levelset_img = image.copy()
		phi, new_phi = levelset(levelset_img, mask)

		return image, phi, new_phi


######### LEVEL SET METHOD ###############
def levelset(levelset_img, mask):
	levelset_img = levelset_img * 255
	levelset_img = levelset_img.astype(np.uint8)
	levelset_img = levelset_img.astype(np.float32)
	phi = mask.copy()

	# Erosion the output mask
	kernel1 = np.array([[0, 0, 1, 0, 0],
				[0, 1, 1, 1, 0],
				[1, 1, 1, 1, 1],
				[0, 1, 1, 1, 0],
				[0, 0, 1, 0, 0]])   
	kernel2 = np.ones((5,5))
	new_phi = binary_erosion(phi, kernel1, iterations = 3)
	phi = 1-phi
	new_phi = 1-new_phi

	# Hyperparameters for level set
	timestep = 1        # time step
	mu = 0.2/timestep   # coefficient of the distance regularization term R(phi)
	iter_inner = 5
	iter_outer = 24
	lmda = 5            # coefficient of the weighted length term L(phi)
	alfa = -3           # coefficient of the weighted area term A(phi)
	epsilon = 1.5       # parameter that specifies the width of the DiracDel
	sigma = 0.8         # scale parameter in Gaussian kernel
	img_smooth = filters.gaussian_filter(levelset_img, sigma) # smooth image by Gaussian convolution
	[Iy, Ix] = np.gradient(img_smooth)
	f = np.square(Ix) + np.square(Iy)
	g = 1 / (1+f)       # edge indicator function.
	potentialFunction = 'double-well'

	# Start level set evolution
	for n in range(iter_outer):
		phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
	iter_refine = 10
	phi = drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_refine, potentialFunction)

	# Preprocessing part with erosion
	for n in range(iter_outer):
		new_phi = drlse_edge(new_phi, g, lmda, mu, alfa, epsilon, 
								timestep, iter_inner, potentialFunction)
	iter_refine = 10
	new_phi = drlse_edge(new_phi, g, lmda, mu, alfa, epsilon, 
							timestep, iter_refine, potentialFunction)
	new_phi = new_phi -np.min(new_phi)
	new_phi = new_phi / np.max(new_phi)
	new_phi = 1-new_phi
	new_phi = (new_phi > 0.6) * 1.0
	kernel2 = np.ones((5,5))
	#         kernel1= np.ones((3,3))
	#         new_phi = binary_erosion(new_phi, kernel2, iterations = 2)
	#         new_phi = binary_dilation(new_phi, kernel2, iterations=1)

	# Preprocessing mask output
	phi = phi -np.min(phi)
	phi = phi / np.max(phi)
	phi = 1-phi
	phi = (phi > 0.6) * 1.0

	# dilation and erosion to remove noise
	kernel = np.array([[0, 0, 1, 0, 0],
						[0, 1, 1, 1, 0],
						[1, 1, 1, 1, 1],
						[0, 1, 1, 1, 0],
						[0, 0, 1, 0, 0]])  
	kernel2 = np.ones((5,5))
	return phi, new_phi