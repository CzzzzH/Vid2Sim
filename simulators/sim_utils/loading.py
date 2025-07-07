import sys
sys.path.append('gs')
from gaussian_renderer import GaussianModel 
from scene.dataset_readers import readCamerasFromTransforms
from scene import cameraList_from_camInfos

import os   
import argparse
import trimesh
import numpy as np
import torch

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor 
from simulators.sim_utils.sampling import farthest_point_sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'gso_000'

class PipelineParamsNoparse:
    """ Same as PipelineParams but without argument parser. """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True # covariances will be updated during simulation
        self.debug = False

def load_cubature_points(points, points_num, random=False): 
    cubature_points, idx = farthest_point_sampling(points.unsqueeze(0), points_num, random_start=random) 
    return cubature_points, idx 

def load_points_from_mesh(path, size=0.5):
    mesh = trimesh.load_mesh(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    v = mesh.vertices
    bmax = v.max(axis=0)
    bmin = v.min(axis=0)
    aabb = bmax - bmin
    center = (bmax + bmin) / 2
    new_v = size * (v - center) / (np.linalg.norm(aabb) * 0.5)
    mesh.vertices = new_v
    return torch.tensor(new_v, device=device, dtype=torch.float32), mesh

def load_model(model_path, sh_degree=3, iteration=-1): 
    gaussians = GaussianModel(sh_degree) 
    gaussians.load_ply(model_path)                                                 
    return gaussians

def load_gaussians(dataset_dir, output_dir, data_name, tag):
    
    if tag == "pred":
        gaussians = load_model(f'{output_dir}/{data_name}/gs_models/pred.ply')
    else:
        gaussians = load_model(f'{output_dir}/{data_name}/gs_models/point_cloud/iteration_30000/point_cloud.ply')
    
    gs_source_path = os.path.abspath(f'{dataset_dir}/{data_name}')
    gs_parser = argparse.ArgumentParser()
    gs_args, _ = gs_parser.parse_known_args()
    gs_args.resolution = 1.0
    gs_args.data_device = device
    gs_pipeline = PipelineParamsNoparse()
    gs_background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # Set white bg
    gs_cams = readCamerasFromTransforms(gs_source_path, "transforms_simulation.json", True, '.png')
    # gs_cams = readCamerasFromTransforms(gs_source_path, "transforms_simulation_nvs.json", True, '.png')
    gs_views = cameraList_from_camInfos(gs_cams, 1.0, gs_args)
    gs_context = {'gs_args': gs_args, 'gs_pipeline': gs_pipeline, 'gs_background': gs_background,
        'gs_cams': gs_cams, 'gs_views': gs_views}
    return gaussians, gs_context

def load_gts(dataset_dir, data_name, view_num=12, frame_num=24):
    images = []
    base_path = f'{dataset_dir}/{data_name}/data'
    for i in range(view_num):
        for j in range(frame_num):
            images.append(pil_to_tensor(Image.open(f'{base_path}/m_{i}_{j}.png')).float()[:3])
    width, height = images[0].shape[-2:]
    images = torch.stack(images).reshape(view_num, frame_num, 3, width, height).to(device) / 255.   
    return images
