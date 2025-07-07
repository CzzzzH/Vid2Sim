import sys
sys.path.append('gs')

import torch 

from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_renderer import render
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def build_covariance_from_scaling_rotation_deformations(scaling, scaling_modifier, rotation, defo_grad=None):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    if defo_grad==None:
        FL = L
    else:
        FL = torch.bmm(defo_grad, L)
    actual_covariance = FL @ FL.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm 

def render_gs(gaussians, views, pipeline, background, dataset_dir, data_name):
    renderings = []
    renderings_bg = []
    for idx in range(len(views)):
        view = views[idx]
        render_out = render(view, gaussians, pipeline, background)
        
        rgb = render_out["render"]
        alpha = render_out["alpha"] 
        bg = pil_to_tensor(Image.open(f'{dataset_dir}/{data_name}/data/r_{idx}_-1.png')).float()[:3]
        bg = bg.to(device) / 255.
        rgb_bg = rgb * alpha + bg * (1 - alpha) # Blend the rendered image with the original background

        renderings.append(rgb)    
        renderings_bg.append(rgb_bg)
        
    return renderings, renderings_bg