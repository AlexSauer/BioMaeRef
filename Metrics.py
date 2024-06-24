from skimage.segmentation import find_boundaries
from skimage import morphology
import numpy as np

def find_cristae_IB_boundary(img):
    bound_inner_IB = find_boundaries(img>1, mode='inner', background=0)
    boundary_cristae_all = find_boundaries(img==2, mode='inner', background=0)
    cristae_IB_boundary = np.logical_and(boundary_cristae_all, bound_inner_IB)
    return cristae_IB_boundary

def find_inner_BG_boundary(img):
    bound_inner_BG = find_boundaries(img>1, mode='inner', background=0)
    boundary_BG_all = find_boundaries(img>0, mode='inner', background=0)
    inner_BG_boundary = np.logical_and(boundary_BG_all, bound_inner_BG)
    return inner_BG_boundary    

def find_inner_boundary(img):
    bound_inner_IB = find_boundaries(img>1, mode='inner', background=0)
    return bound_inner_IB

def find_holes(img):
    holes_filled = morphology.remove_small_holes(img>1, area_threshold=10)
    holes = np.logical_and(holes_filled>0, img<=1)
    return holes