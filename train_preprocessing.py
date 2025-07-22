import argparse
import random
import json
import os

from omegaconf import OmegaConf

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset_dir", type=str, default="dataset/objaverse") 
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    random.seed(0) 
    data_list = []
    
    with open(f"{args.dataset_dir}/objaverse_uid_list.json", "r") as f:
        obj_list = json.load(f)

    for obj_path in obj_list:

        obj_id = obj_path.split('/')[-1]

        if not os.path.exists(f'{args.dataset_dir}/outputs/{obj_id}/renderings/015.png'):
            continue
        
        gt_phys_params = OmegaConf.load(f'{args.dataset_dir}/outputs/{obj_id}/gt_phys_params.yaml')
        data_info = {}
        data_info['obj_id'] = obj_id
        data_info['yms'] = gt_phys_params['yms']
        data_info['prs'] = gt_phys_params['prs']
        data_list.append(data_info)
    
    random.shuffle(data_list)
    split_idx = int(len(data_list) * (1 - args.val_ratio))
    train_data_list = data_list[:split_idx]
    val_data_list = data_list[split_idx:]

    with open(f'{args.dataset_dir}/objaverse_train_list.json', 'w') as f:
        json.dump(train_data_list, f)
    with open(f'{args.dataset_dir}/objaverse_val_list.json', 'w') as f:
        json.dump(val_data_list, f)