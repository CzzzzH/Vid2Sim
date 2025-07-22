import sys
sys.path.append('./')  

import argparse
import os 
import json
import torch 
import numpy as np
import gc
import time 
 
from omegaconf import OmegaConf 
from utils.seeding import seed_everything 
from simulators.lbs_simulator import LBSSimulator
from simulators.sim_utils.loading import load_points_from_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# For data augmentation (not used yet)
def rot(points, center, angles):

    if angles == None:
        return points

    angle_x, angle_y, angle_z = angles
    angle_x = torch.tensor(angle_x, device=device, dtype=points.dtype)
    angle_y = torch.tensor(angle_y, device=device, dtype=points.dtype)
    angle_z = torch.tensor(angle_z, device=device, dtype=points.dtype)

    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

    rot_x = torch.tensor([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]], device=device, dtype=points.dtype)

    rot_y = torch.tensor([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]], device=device, dtype=points.dtype)

    rot_z = torch.tensor([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]], device=device, dtype=points.dtype)

    rotation_matrix = rot_z @ rot_y @ rot_x
    rotated_points = torch.matmul(points, rotation_matrix.T)
    return rotated_points

def simulate_data(args):

    base_dir = f'{args.dataset_dir}/raw/hf-objaverse-v1/glbs'

    with open(f"{args.dataset_dir}/objaverse_uid_list.json", "r") as f:
        obj_list = json.load(f)
    start_idx = max(args.start_idx, 0)
    end_idx = min(args.end_idx, len(obj_list))
    output_dir = f'{args.dataset_dir}/outputs'
    print(f'Simulating {end_idx - start_idx} objects from {start_idx} to {end_idx}')

    for i, obj_path in enumerate(obj_list[start_idx:end_idx]): 

        obj_id = obj_path.split('/')[-1] 
        seed_everything(obj_id)
        
        if os.path.exists(f'{output_dir}/{obj_id}/meshes/015.glb'):
            continue
        
        gc.collect()
        torch.cuda.empty_cache() 
        os.makedirs(f'{output_dir}/{obj_id}/meshes', exist_ok=True)  
        start_time = time.time()
        
        min_logs_yms, max_logs_yms = 4.0, 6.0
        min_prs, max_prs = 0.2, 0.5
        gt_config = OmegaConf.create()
        gt_config.yms = float(np.power(10, np.random.uniform(min_logs_yms, max_logs_yms))) 
        gt_config.prs = float(np.random.uniform(min_prs, max_prs))

        with open(f'{output_dir}/{obj_id}/gt_phys_params.yaml', 'w') as f:
            OmegaConf.save(gt_config, f)
 
        sim_args = OmegaConf.load(args.config)
        sim_args.yms = gt_config.yms
        sim_args.prs = gt_config.prs
        sim_args.tag = 'base' 

        os.system(f'cp {base_dir}/{obj_path}.glb {output_dir}/{obj_id}/mesh.glb')
        points, mesh = load_points_from_mesh(f'{output_dir}/{obj_id}/mesh.glb') 
        sim_args.num_cubature_points = min(points.shape[0], 500)
        
        simulator = LBSSimulator(sim_args, output_dir, output_dir, obj_id, produce_data=True)
        simulator.set_material()
        simulator.lbs_model.load_state_dict(torch.load(args.ckpt_lbs))
        simulator.refine_lbs(skip_lbs=True)
        torch.save(simulator.lbs_model.state_dict(), f'{output_dir}/{obj_id}/model_base.pth')

        simulator.initialize_simulator()
        simulator.simulate_fast_forward(target_step=15, render=True)  
        for k in range(len(simulator.save_list)):
            mesh.vertices = simulator.save_list[k]
            mesh.export(f'{output_dir}/{obj_id}/meshes/{k:03d}.glb')
        print(f'Success to simulate {obj_id} in {time.time() - start_time:.2f}s')            

def process_data_list(args):

    base_dir = f'{args.dataset_dir}/raw/hf-objaverse-v1/glbs'
    num_data = 50000 # Assume we use the first 50k data
    uids = []

    # Walk through the base_dir and print all file paths
    for root, dirs, files in os.walk(base_dir):
        for file in files: 
            uids.append(f'{root.split("/")[-1]}/{file.split(".")[0]}')
            if len(uids) >= num_data:
                break
        if len(uids) >= num_data:
            break
    
    with open(f'{args.dataset_dir}/objaverse_uid_list.json', 'w') as f:
        json.dump(uids, f)
    print(f'Add {len(uids)} objects and saved to {args.dataset_dir}/objaverse_uid_list.json')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/objaverse")
    parser.add_argument("--config", type=str, default="config/objaverse.yaml") 
    parser.add_argument("--ckpt_lbs", type=str, default="checkpoints/ckpt_lbs_template.pth")
    parser.add_argument("--task", type=str, default="process")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50000)
    args = parser.parse_args()

    if args.task == 'process':
        process_data_list(args)
    elif args.task == 'simulate':
        simulate_data(args)
    else:
        raise ValueError(f'Invalid task: {args.task}')

