import numpy as np

T = 0.6  # threshold to take binary, consider to be changed

def binarize_phi(phi):
    '''
    Convert messy mask after levelset to a binary mask
    :param phi: <numpy array [H, W]> masked after processed by levelset
    '''
    binary_phi = phi.copy()
    binary_phi = binary_phi - np.min(binary_phi)
    binary_phi = binary_phi / np.max(binary_phi)
    binary_phi = 1 - binary_phi
    binary_phi = (binary_phi > T) * 1.0
    return binary_phi


def CustomBinarize(inputImg, commonThreshold=None):
    """Binarize iamge
    im: hessian filtered OCTA image, im (0, 255)
    
    Returns:
        BW: black and white skeleton
    """
    boxSize = 5
    hs = int((boxSize - 1)/2)
    H, W = inputImg.shape
    
    if commonThreshold is None:
        commonThreshold = np.sum(inputImg)/(W*H)
    commonThreshold = 20 # ???
    
    #
    CH = np.ones((H,W))*commonThreshold
    for x in range(hs + 1, W - hs -1):
        for y in range(hs + 1, H - hs - 1):
            if inputImg[x, y] > commonThreshold:
                cth = 0
                for i in range(-hs, hs):
                    for j in range(-hs, hs):
                        cth = cth + inputImg[x + i, y + j]
                cth = cth/(boxSize**2)
                CH[x, y] = cth

    #
    for x in range(W):
        for y in range(H):
            if inputImg[x, y] > CH[x,y]:
                inputImg[x, y] = 255
            else:
                inputImg[x, y] = 0

    BW = inputImg
    return BW