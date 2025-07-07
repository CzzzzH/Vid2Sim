import torch
import kaolin.physics as physics
import torch.nn.functional as F

from functools import partial
from kaolin.physics.simplicits.losses import loss_ortho
from kaolin.physics.simplicits.utils import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac
from kaolin.physics.materials import neohookean_elastic_material, to_lame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def potential_sum(output, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_fcns = [], pt_wise_fcns = []):
    # updates the quantity calculated in the output value
    F_ele = torch.matmul(dFdz, z) + bigI
    F_ele.requires_grad_(True)
    x = (B @ z + x0_flat).reshape(-1,3)
    for e in defo_grad_fcns:
        output += e(F_ele)
    for e in pt_wise_fcns:
        output += e(x)

def energy_p(z, B, x0_flat, model, defo_grad_fcns, pt_wise_fcns):
    output = torch.tensor([0], device=device, dtype=torch.float32)
    points = x0_flat.reshape(-1, 3)
    x = (B @ z + x0_flat).reshape(-1,3)
    partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=z.reshape(-1,3,4).unsqueeze(0), fcn=model)
    F_z = finite_diff_jac(partial_weight_fcn_lbs, points)
    for e in defo_grad_fcns:
        output = output + e(F_z)
    for e in pt_wise_fcns:
        output = output + e(x)
    return output[0]

def newton_E(z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, model,
    defo_grad_energies = [], pt_wise_energies = []):
    E = torch.tensor([0], device=device, dtype=torch.float32)
    potential_sum(E, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_energies, pt_wise_energies) 
    return 0.5 * z.T @ BMB @ z - z.T @ BMB @ z_prev - dt * z.T @ BMB @ z_dot + dt * dt * E

def newton_G(z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, model,
        defo_grad_energies = [], pt_wise_energies = [], defo_grad_gradients = [], pt_wise_gradients = []):
    G = torch.zeros_like(z)
    potential_sum(G, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_gradients, pt_wise_gradients)
    return BMB @ z - BMB @ z_prev - dt * BMB @ z_dot + dt * dt * G

def newton_H(z, z_prev, z_dot, B, BMB, dt, x0_flat, dFdz, bigI, model,
    defo_grad_energies = [], pt_wise_energies = [], defo_grad_hessians = [], pt_wise_hessians = []):
    H = torch.zeros(z.shape[0], z.shape[0], device=device, dtype=torch.float32)
    potential_sum(H, z, z_dot, B, dFdz, x0_flat, bigI, defo_grad_hessians, pt_wise_hessians)
    return BMB + dt * dt * H

def loss_elastic(model, pts, transforms, appx_vol, yms, prs, interp_step):
    mus, lams = to_lame(yms, prs)
    partial_weight_fcn_lbs = partial(weight_function_lbs,  tfms = transforms, fcn = model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:,:,0]
    N, B = pt_wise_Fs.shape[0:2]
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)

    # We only use neohookean constitutive model for data-free initialization here
    
    # lin_elastic = (1-interp_step) * linear_elastic_material.linear_elastic_energy(mus, lams, pt_wise_Fs)
    # neo_elastic = (interp_step) * neohookean_elastic_material.neohookean_energy(mus, lams, pt_wise_Fs)
    # return (appx_vol / pts.shape[0])*(torch.sum(lin_elastic + neo_elastic))
    energy = neohookean_elastic_material.neohookean_energy(mus, lams, pt_wise_Fs)
    return (appx_vol / pts.shape[0]) * (torch.sum(energy))

def train_step_lbs(model, normalized_pts, en_interp, batch_size, num_handles,
    appx_vol, yms, prs, num_samples, le_coeff, lo_coeff):
    
    batch_transforms = 0.1 * torch.randn(batch_size, num_handles, 3, 4,
        dtype=normalized_pts.dtype, device=normalized_pts.device)

    sample_indices = torch.randperm(normalized_pts.shape[0], device=normalized_pts.device)[:num_samples]
    sample_pts = normalized_pts[sample_indices] 
    sample_yms = yms[sample_indices]
    sample_prs = prs[sample_indices]
    weights = model(sample_pts)
    le = le_coeff * loss_elastic(model, sample_pts, batch_transforms, appx_vol, sample_yms, sample_prs, en_interp)
    lo = lo_coeff * loss_ortho(weights)

    return le, lo

def train_step_jacobian(model, j_model, normalized_pts, batch_size, num_handles, num_samples):
    
    z = 0.1 * torch.randn(batch_size, num_handles, 3, 4, dtype=normalized_pts.dtype, device=normalized_pts.device)
    sample_indices = torch.randperm(normalized_pts.shape[0], device=normalized_pts.device)[:num_samples]
    sample_pts = normalized_pts[sample_indices] 
    model_rigid = lambda x: torch.cat((model(x), torch.ones((x.shape[0], 1), device=device)), dim=1)
    
    bigI = torch.tile(torch.eye(3, device=device).flatten().unsqueeze(dim=1), (sample_pts.shape[0] * batch_size, 1))
    partial_weight_fcn_lbs = partial(physics.simplicits.utils.weight_function_lbs, tfms=z, fcn=model_rigid)
    F_ref = finite_diff_jac(partial_weight_fcn_lbs, sample_pts, eps=1e-7).reshape(-1, 9)
    dFdz_pred = j_model(x=sample_pts).view(sample_pts.shape[0], 9, -1)
    batch_dFdz_pred = dFdz_pred.unsqueeze(1).repeat(1, batch_size, 1, 1).reshape(sample_pts.shape[0] * batch_size, 9, -1)
    batch_z = z.unsqueeze(0).repeat(sample_pts.shape[0], 1, 1, 1, 1).reshape(sample_pts.shape[0] * batch_size, -1, 1)
    F_pred = torch.bmm(batch_dFdz_pred, batch_z).squeeze(-1) + bigI.reshape(-1, 9)
    lf = F.l1_loss(F_pred, F_ref)

    return lf