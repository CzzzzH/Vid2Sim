import sys
import os
import json
import argparse
import random
import imageio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import tyro 
 
from tqdm import trange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import VideoMAEModel
from safetensors.torch import load_file
from PIL import Image

from utils.seeding import seed_everything
from models.phys_predictor import SimulationDataset, FeedForwardPredictor, RegressionHead, LBSHead
from models.lbs_networks import SimplicitsMLP

sys.path.append('LGM')
from LGM.core.models import LGM
from LGM.core.options import AllConfigs  
from kiui.op import recenter

sys.path.append('gs')
from gs.train import training
from gs.arguments import ModelParams, OptimizationParams, PipelineParams
from gs.utils.general_utils import safe_state 

from simulators.lbs_simulator import LBSSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_phys_params(args):

    # res = 224
    res = 448
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    data_name = args.data_name
    os.makedirs(f'{output_dir}/{data_name}', exist_ok=True)
    
    pred_dataset = SimulationDataset(dataset_dir, data_name, res=res, frame_num=16)
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, num_workers=16, shuffle=False) 
    backbone_model_pretrained = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

    # Interpolate position embeddings
    if res != 224:
        backbone_model_pretrained.embeddings.patch_embeddings.image_size = (res, res)
        pos_tokens = backbone_model_pretrained.embeddings.position_embeddings
        T = 8
        P = int((pos_tokens.shape[1] // T) ** 0.5)
        C = pos_tokens.shape[2]
        new_P = res // 16
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, T, P, P, C)
        pos_tokens = pos_tokens.reshape(-1, P, P, C).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_P, new_P), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
        backbone_model_pretrained.embeddings.position_embeddings = pos_tokens  # update

    predictor = FeedForwardPredictor(backbone_model_pretrained).to(device)
    predictor.load_state_dict(torch.load(args.ckpt_predictor).state_dict())
    predictor.eval() 
    
    # It only predicts parameters for one object here (you can change it to batch prediction)
    for data in pred_dataloader:
    
        video = data['video'].to(device) 
        output = predictor(video)
        yms_pred, prs_pred, lbs_pred = output['yms'], output['prs'], output['lbs']
  
        print(f"[[Stage I]] Predict physical parameters for {data_name}:")
        print(f"[[Stage I]] Young's Modulus={torch.pow(10, yms_pred).item()} \t Poisson's Ratio={prs_pred.item()}")
        
        params_path = f'{output_dir}/{data_name}/init_params.yaml'
        params = OmegaConf.create()
        params.init_yms = torch.pow(10, yms_pred).item()
        params.init_prs = prs_pred.item() 
        OmegaConf.save(params, params_path)
      
        # Assign predicted LBS weights and biases
        mlp_predict = SimplicitsMLP(spatial_dimensions=3, layer_width=64, num_handles=10, num_layers=8)
        mlp_predict.load_state_dict(torch.load(args.ckpt_lbs))
        mlp_predict.net[-1].weight.data = lbs_pred[0:1, :640].reshape(mlp_predict.net[-1].weight.data.shape)
        mlp_predict.net[-1].bias.data = lbs_pred[0:1, 640:].reshape(mlp_predict.net[-1].bias.data.shape)
        os.makedirs(f'{output_dir}/{data_name}/models', exist_ok=True)
        torch.save(mlp_predict.state_dict(), f'{output_dir}/{data_name}/models/model_pred.pth')

# Modified from LGM/infer.py
def predict_gs_LGM(args): 

    print(f"[[Stage I]] Predict GS for {args.data_name}...") 
     
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    sys.argv = ['example.py', 'big']
    opt = tyro.cli(AllConfigs)

    model = LGM(opt)
    ckpt = load_file(args.ckpt_lgm, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.half().to(device)
    model.eval()
    # bg_remover = rembg.new_session()
    
    rays_embeddings = model.prepare_default_rays(device)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    for view_idx in range(4): 
        
        # We directly use the segmented image here (if your own dataset is not segmented,
        # you can use the rembg (the commented line) or Segment-Anything to remove the background)

        input_image = Image.open(f'{args.dataset_dir}/{args.data_name}/data/a_{view_idx}_0.png')
        input_image = np.array(input_image) 
        # input_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
        mask = input_image[..., -1] > 0

        # Center the masked object in the image
        H, W = input_image.shape[:2]
        coords = np.nonzero(mask)
        if len(coords[0]) > 0:  # Check if mask is not empty
            x_min, x_max, y_min, y_max = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max() 
            mask_center_x, mask_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            img_center_x, img_center_y = H // 2, W // 2 

            shift_x, shift_y = img_center_x - mask_center_x, img_center_y - mask_center_y 
            centered_image = np.zeros_like(input_image)
            centered_mask = np.zeros_like(mask) 

            for i in range(H):
                for j in range(W):
                    new_i = i + shift_x
                    new_j = j + shift_y
                    if 0 <= new_i < H and 0 <= new_j < W:
                        centered_image[new_i, new_j] = input_image[i, j]
                        centered_mask[new_i, new_j] = mask[i, j]
            input_image = centered_image
            mask = centered_mask

        image = recenter(input_image, mask, border_ratio=0.2) # original LGM operator
        image = image.astype(np.float32) / 255.0  
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        images.append(image)

    mv_image = np.stack(images, axis=0)
    
    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians 
        os.makedirs(f'{args.output_dir}/{args.data_name}/gs_models', exist_ok=True)
        model.gs.save_ply(gaussians, f'{args.output_dir}/{args.data_name}/gs_models/pred.ply')

# Modified from gs/train.py
def refine_gs(args):
    
    print(f"[[Stage II]] Refine GS for {args.data_name} (using standard 3DGS training) ...")  

    gs_parser = argparse.ArgumentParser()
    lp = ModelParams(gs_parser)
    op = OptimizationParams(gs_parser)
    pp = PipelineParams(gs_parser)
    gs_parser.add_argument('--ip', type=str, default="127.0.0.1")
    gs_parser.add_argument('--port', type=int, default=6009)
    gs_parser.add_argument('--debug_from', type=int, default=-1)
    gs_parser.add_argument('--detect_anomaly', action='store_true', default=False)
    gs_parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    gs_parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    gs_parser.add_argument("--quiet", action="store_true")
    gs_parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    gs_parser.add_argument("--start_checkpoint", type=str, default = None)
    gs_args = gs_parser.parse_args(["-s", f'{args.dataset_dir}/{args.data_name}',
                                    "-m", f'{args.output_dir}/{args.data_name}/gs_models',
                                    "--white_background"])
    gs_args.save_iterations.append(gs_args.iterations)
     
    # Initialize system state (RNG)
    safe_state(gs_args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(gs_args.detect_anomaly)
    training(lp.extract(gs_args), op.extract(gs_args), pp.extract(gs_args), gs_args.test_iterations,
        gs_args.save_iterations, gs_args.checkpoint_iterations, gs_args.start_checkpoint, gs_args.debug_from)
 
def refine_lbs(args):

    print(f"[[Stage II]] Refine Neural LBS & Jacobian for {args.data_name} (using data-free training)") 
    with open(f'{args.output_dir}/{args.data_name}/init_params.yaml', 'r') as f:
        init_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args.tag = 'base' 
    sim_args.yms = init_params.init_yms
    sim_args.prs = init_params.init_prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material() 
    simulator.refine_lbs()

    # (Optional) Check the simulation results at this stage
    # simulator.load_lbs()
    # simulator.initialize_simulator()
    # simulator.simulate_fast_forward(target_step=15, view_indices=simulator.total_view_indices, render=True)
    # simulator.save_images(simulator.tag)
    # psnr, ssim = simulator.calculate_metrics(view_indices=simulator.total_view_indices, end_step=16)
    # print(f"After refinement -- PSNR: {psnr.item()}, SSIM: {ssim.item()}")

def joint_optimization(args):

    print(f"[[Stage II]] Joint optimization for {args.data_name}")
    with open(f'{args.output_dir}/{args.data_name}/init_params.yaml', 'r') as f:
        init_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args.tag = 'base' 
    sim_args.yms = init_params.init_yms
    sim_args.prs = init_params.init_prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name, optimization=True)
    simulator.set_material() 
    simulator.load_lbs()

    record = {}
    record['train_list'] = []
    record['test_list'] = []
    best_params = OmegaConf.create()

    # You can try using less optimization iterations and it's sufficient to get a good result
    pbar = trange(sim_args.optimization_iters + 1)
    best_psnr = 0
    best_ssim = 0

    for iter in pbar: 
        
        simulator.initialize_simulator() # Need to re-initialize every iteration since the lbs parameters are updated 
        start_step = random.randint(sim_args.simulation_start_step, 11)
        end_step = start_step + 4

        if iter % sim_args.optimization_checkpoint_interval == 0: # Validate the model for dynamic reconstruction
            simulator.simulate_fast_forward(15, simulator.total_view_indices, render=True)  
            psnr, ssim = simulator.calculate_metrics(simulator.total_view_indices, end_step=16)
            print(f"[Iter {iter}: PSNR={psnr} SSIM={ssim}")
            yms_pred, prs_pred = torch.pow(10, simulator.pred_yms_normalized), simulator.pred_prs_normalized 
            record['test_list'].append({'psnr': psnr.item(), 'ssim': ssim.item(), 'yms': yms_pred.item(), 'prs': prs_pred.item()})
            if psnr > best_psnr and ssim > best_ssim: # Save the best model
                best_psnr, best_ssim = psnr, ssim
                record['best_psnr'], record['best_ssim'] = psnr.item(), ssim.item()
                best_params.yms, best_params.prs = yms_pred.item(), prs_pred.item()
                OmegaConf.save(best_params, f'{args.output_dir}/{args.data_name}/best_params.yaml')
                torch.save(simulator.lbs_model.state_dict(), f'{args.output_dir}/{args.data_name}/models/model_best.pth')
                torch.save(simulator.jacobian_model.state_dict(), f'{args.output_dir}/{args.data_name}/models/jmodel_best.pth') 
            simulator.reset_simulator()
            
        if iter != sim_args.optimization_iters:
            view_indices = torch.arange(0, len(simulator.gs_context['gs_views']), device=device, dtype=torch.long)
            simulator.simulate_fast_forward(start_step, view_indices)
            simulator.simulate_with_grad(end_step, view_indices)
            rendering_loss = simulator.update_parameters(view_indices, start_step, end_step)
            yms_pred, prs_pred = torch.pow(10, simulator.pred_yms_normalized), simulator.pred_prs_normalized
            record['train_list'].append({'loss': rendering_loss.item(), 'yms': yms_pred.item(), 'prs': prs_pred.item()})
            
        pbar.set_description(f"Loss={rendering_loss.item()} E={yms_pred.item()} ν={prs_pred.item()} ")
        pbar.set_postfix(lr=float(simulator.mat_optimizer.param_groups[0]['lr']))
        with open(f'{args.output_dir}/{args.data_name}/optimization_record.json', 'w') as f:
            json.dump(record, f)

def final_simulation(args):

    print(f"[[Stage II]] Final simulation for {args.data_name}")
    with open(f'{args.output_dir}/{args.data_name}/best_params.yaml', 'r') as f:
        best_params = OmegaConf.load(f)
    sim_args = OmegaConf.load(args.config) 
    sim_args.tag = 'best' 
    sim_args.yms = best_params.yms
    sim_args.prs = best_params.prs 
    simulator = LBSSimulator(sim_args, args.dataset_dir, args.output_dir, args.data_name)
    simulator.set_material() 
    simulator.load_lbs()
    simulator.initialize_simulator()

    # For future state prediction, you can increase the target step
    simulator.simulate_fast_forward(target_step=15, view_indices=simulator.total_view_indices, render=True)
    simulator.save_images(simulator.tag)
    psnr, ssim = simulator.calculate_metrics(view_indices=simulator.total_view_indices, end_step=16)
    print(f"After joint optimization -- PSNR: {psnr.item()}, SSIM: {ssim.item()}")

    # Visualize the simulation results at front view
    output_dir = f'{args.output_dir}/{args.data_name}/render_best'
    gt_dir = f'{args.dataset_dir}/{args.data_name}/data'
    pred_imgs, gt_imgs = [], []
    for i in range(16): 
        pred_imgs.append(imageio.imread(f'{output_dir}/r_0_{i}.png'))
        gt_imgs.append(imageio.imread(f'{gt_dir}/r_0_{i}.png'))
    pred_imgs = np.stack(pred_imgs, axis=0)
    gt_imgs = np.stack(gt_imgs, axis=0) 
    imageio.mimsave(f'{args.output_dir}/{args.data_name}/recon.gif', pred_imgs, fps=20, loop=0)
    imageio.mimsave(f'{args.output_dir}/{args.data_name}/gt.gif', gt_imgs, fps=20, loop=0)

def run_recon(args):

    ### [Stage I] Predict initial physical parameters from single view video & reconstruct GS with pretrained LGM 
    predict_phys_params(args)
    predict_gs_LGM(args)

    ### [Stage II] Refine GS, Neural LBS and Neural Jacobian before joint optimization
    refine_gs(args)
    refine_lbs(args)

    ### [Stage II] Joint optimization
    joint_optimization(args)
    final_simulation(args)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gso.yaml") 
    parser.add_argument("--dataset_dir", type=str, default="dataset/GSO")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_name", type=str, default="bus")
    parser.add_argument("--ckpt_predictor", type=str, default="checkpoints/ckpt_phys_predictor.pth")
    parser.add_argument("--ckpt_lbs", type=str, default="checkpoints/ckpt_lbs_template.pth")
    parser.add_argument("--ckpt_lgm", type=str, default="checkpoints/ckpt_lgm.safetensors")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
 
    seed_everything(args.seed)
    run_recon(args)