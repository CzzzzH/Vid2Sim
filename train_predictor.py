import sys
sys.path.append('./')   

import torch
import numpy as np
import os 
import wandb
import argparse

from torch import nn
from torch.utils.data import DataLoader
from transformers import VideoMAEModel
from tqdm import tqdm
from utils.seeding import seed_everything
from models.phys_predictor import SimulationDataset, FeedForwardPredictor 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train_predictor(args):
    
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
        wandb.login()
        wandb.init(project="simulator-feedforward")
 
    train_dataset = SimulationDataset(args.dataset_dir, res=args.res, frame_num=16, split='train')
    val_dataset = SimulationDataset(args.dataset_dir, res=args.res, frame_num=16, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    backbone_model_pretrained = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    
    # Interpolate position embeddings
    if args.res != 224:
        backbone_model_pretrained.embeddings.patch_embeddings.image_size = (args.res, args.res)
        pos_tokens = backbone_model_pretrained.embeddings.position_embeddings
        T = 8
        P = int((pos_tokens.shape[1] // T) ** 0.5)
        C = pos_tokens.shape[2]
        new_P = args.res // 16
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
        pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_P, new_P), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        backbone_model_pretrained.embeddings.position_embeddings = pos_tokens  # update

    predictor = FeedForwardPredictor(backbone_model_pretrained, in_features=768).to(device)
    if args.resume_path is not None and os.path.exists(args.resume_path):
        predictor.load_state_dict(torch.load(args.resume_path))

    loss_fn = nn.MSELoss()  
    optimizer = torch.optim.Adam([{'params': predictor.backbone.parameters(), 'lr': args.lr},
                                   {'params': predictor.regression_head_yms.parameters(), 'lr': args.lr},
                                   {'params': predictor.regression_head_prs.parameters(), 'lr': args.lr},
                                   {'params': predictor.lbs_head.parameters(), 'lr': args.lr}
                                   ], betas=(0.9, 0.98))
    
    os.makedirs(f'{args.save_dir}/{args.task_tag}', exist_ok=True) 
    predictor.train()
    total_iteration = 1

    for epoch in range(1, args.total_epochs + 1):

        pbar = tqdm(train_dataloader)
        
        for data in pbar:
            
            video = data['video'].to(device)
            yms_gt = data['yms_gt'].to(device)
            prs_gt = data['prs_gt'].to(device)
            lbs_gt = data['lbs_gt'].to(device)
            
            output = predictor(video)
            yms_pred, prs_pred, lbs_pred = output['yms'], output['prs'], output['lbs']
            
            parameter_loss = loss_fn(yms_pred, yms_gt) + 100 * loss_fn(prs_pred, prs_gt)
            lbs_loss = loss_fn(lbs_pred, lbs_gt)
            loss = parameter_loss + lbs_loss

            pbar.set_description(f'Epoch {epoch}, Total Iteration {total_iteration}, Loss: {parameter_loss.item()}')
            if args.use_wandb:
                wandb.log({'train_loss': loss.item()})

            optimizer.zero_grad()   
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Validation
            if total_iteration % args.val_interval == 0:
                
                torch.save(predictor , f'{args.save_dir}/{args.task_tag}/ckpt_{total_iteration:06d}.pth')

                print("Starting Validation")
                predictor.eval()
                valid_losses = []
                
                for i, data in enumerate(val_dataloader):

                    video = data['video'].to(device)
                    yms_gt = data['yms_gt'].to(device)
                    prs_gt = data['prs_gt'].to(device)
                    lbs_gt = data['lbs_gt'].to(device) 
                    
                    output = predictor(video)
                    yms_pred, prs_pred, lbs_pred = output['yms'], output['prs'], output['lbs']
                    lbs_loss = loss_fn(lbs_pred, lbs_gt)
                    loss = parameter_loss + lbs_loss
                    valid_losses.append(loss.item())
                    
                    print(f'Total iteration {total_iteration}, Loss: {loss.item()}, LBS Loss: {lbs_loss.item()} \
                        yms_gt: {torch.pow(10, yms_gt).item()}, yms_pred: {torch.pow(10, yms_pred).item()}, \
                        prs_gt: {prs_gt.item()}, prs_pred: {prs_pred.item()}')
                
                valid_losses_mean = np.mean(valid_losses)
                if args.use_wandb:
                    wandb.log({f'val_loss_{args.task_tag}': valid_losses_mean})
                    wandb.log({f'val_loss': valid_losses_mean})
                
                predictor.train()
                             
            # Save
            if total_iteration % args.save_interval == 0:
                torch.save(predictor , f'{args.save_dir}/{args.task_tag}/ckpt_{total_iteration:06d}.pth')

            total_iteration += 1

    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset_dir", type=str, default="dataset/objaverse") 
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--task_tag", type=str, default="phys_predictor")
    parser.add_argument("--res", type=int, default=448) 
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--total_epochs", type=int, default=5) 
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5) 
    parser.add_argument("--resume_path", type=str, default=None)
    args = parser.parse_args()

    seed_everything(0)

    train_predictor(args)