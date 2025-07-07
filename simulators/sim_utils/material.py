import torch 
import kaolin.physics.utils as physics_utils
from kaolin.physics.materials import neohookean_elastic_material
from kaolin.physics.materials import to_lame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def projection_plasticine(F, yield_stress, mu):
    U, Sig, V = torch.svd(F)  
    V_T = V.transpose(1, 2)
    # Sig = torch.clamp(Sig, max=0.05)
    eps = torch.log(Sig)
    eps_hat = eps - torch.mean(eps, dim=1, keepdim=True)
    eps_hat_norm = torch.norm(eps_hat, dim=1, keepdim=True)
    delta_gamma = eps_hat_norm - yield_stress / (2 * mu)
    F = torch.where(delta_gamma.unsqueeze(-1) > 0, U @ torch.diag_embed(torch.exp(eps - delta_gamma * eps_hat / eps_hat_norm)) @ V_T, F)
    return F

def projection_sand(F, friction, mu, lam):
    
    U, Sig, V = torch.svd(F)  
    V_T = V.transpose(1, 2)
    # Sig = torch.clamp(Sig, max=0.05)
    alpha = torch.sqrt(torch.tensor(2 / 3, device=friction.device)) * (2 * torch.sin(friction) / (3 - torch.sin(friction)))
    eps = torch.log(Sig)
    trace_eps = torch.sum(eps, dim=1, keepdim=True)
    eps_hat = eps - torch.mean(eps, dim=1, keepdim=True)
    eps_hat_norm = torch.norm(eps_hat, dim=1, keepdim=True)
    delta_gamma = eps_hat_norm + alpha * (3 * lam + 2 * mu) * trace_eps / (2 * mu)
    F_out = U @ torch.diag_embed(torch.exp(eps - delta_gamma * eps_hat / eps_hat_norm)) @ V_T
    F_out = torch.where(delta_gamma.unsqueeze(-1) > 0, U @ V_T, F_out)
    F_out = torch.where(torch.logical_and(delta_gamma.unsqueeze(-1) <= 0, trace_eps.unsqueeze(-1) <= 0), F, F_out)
    return F_out

class NeohookeanMaterial(physics_utils.ForceWrapper):

    def __init__(self, yms, prs): 
        self.mus, self.lams = to_lame(yms, prs) 

    def _energy(self, defo_grad):
        return neohookean_elastic_material.unbatched_neohookean_energy(self.mus, self.lams, defo_grad)
    
    def _gradient(self, defo_grad):
        return neohookean_elastic_material.unbatched_neohookean_gradient(self.mus, self.lams, defo_grad)
    
    def _hessian(self, defo_grad):
        return neohookean_elastic_material.unbatched_neohookean_hessian(self.mus, self.lams, defo_grad)

class PlasticineMaterial(physics_utils.ForceWrapper):

    def __init__(self, yms, prs):
        
        # Use a fixed value for an example
        yms = torch.ones_like(yms) * 1e6
        prs = torch.ones_like(prs) * 0.3 
        self.yield_stress = torch.ones_like(yms) * 5 * 1e2
        self.mus, self.lams = to_lame(yms, prs)
        self.I = torch.eye(3).to(device).unsqueeze(0).repeat(self.mus.shape[0], 1, 1)

    def _energy(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_plasticine(F, self.yield_stress, mu)
        
        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        G_2 = G @ G

        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        trace_G_2 = torch.einsum('bii->b', G_2).unsqueeze(1)
        E = 0.5 * lam * (trace_G ** 2) + mu * trace_G_2
        return E
    
    def _gradient(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_plasticine(F, self.yield_stress, mu)

        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        S = lam.unsqueeze(-1) * trace_G.unsqueeze(-1) * self.I + 2 * mu.unsqueeze(-1) * G
        P = F @ S         
        return P
    
    def _hessian(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_plasticine(F, self.yield_stress, mu)
        
        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        S = lam.unsqueeze(-1) * trace_G.unsqueeze(-1) * self.I + 2 * mu.unsqueeze(-1) * G
        P = F @ S

        H = torch.zeros(self.mus.shape[0], 9, 9, device=self.mus.device) 
        for i in range(3):
            for j in range(3):
                grad_ij = torch.autograd.grad(P[:, i, j].sum(), F, retain_graph=True)[0]
                H[:, 3 * i + j, :] = grad_ij.reshape(self.mus.shape[0], -1)
        return H
    
class SandMaterial(physics_utils.ForceWrapper):

    def __init__(self, yms, prs):
        
        # Use a fixed value for an example
        yms = torch.ones_like(yms) * 1e5
        prs = torch.ones_like(prs) * 0.3 
        self.friction = torch.ones_like(yms) * (10 * torch.pi / 180)
        self.mus, self.lams = to_lame(yms, prs)
        self.I = torch.eye(3).to(device).unsqueeze(0).repeat(self.mus.shape[0], 1, 1)

    def _energy(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_sand(F, self.friction, mu, lam)
        
        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        G_2 = G @ G

        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        trace_G_2 = torch.einsum('bii->b', G_2).unsqueeze(1)
        E = 0.5 * lam * (trace_G ** 2) + mu * trace_G_2
        return E
    
    def _gradient(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_sand(F, self.friction, mu, lam)
        
        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        S = lam.unsqueeze(-1) * trace_G.unsqueeze(-1) * self.I + 2 * mu.unsqueeze(-1) * G
        P = F @ S         
        return P
    
    def _hessian(self, defo_grad):
        
        mu = self.mus.reshape(-1, 1)
        lam = self.lams.reshape(-1, 1)
        F = defo_grad.reshape(-1, 3, 3)
        F = projection_sand(F, self.friction, mu, lam)
        
        F_T = F.transpose(1, 2)
        G = 0.5 * (F_T @ F - self.I)
        trace_G = torch.einsum('bii->b', G).unsqueeze(1)
        S = lam.unsqueeze(-1) * trace_G.unsqueeze(-1) * self.I + 2 * mu.unsqueeze(-1) * G
        P = F @ S

        H = torch.zeros(self.mus.shape[0], 9, 9, device=self.mus.device) 
        for i in range(3):
            for j in range(3):
                grad_ij = torch.autograd.grad(P[:, i, j].sum(), F, retain_graph=True)[0]
                H[:, 3 * i + j, :] = grad_ij.reshape(self.mus.shape[0], -1)
        return H

# Moving floor boundary condition
class MovingFloor(physics_utils.ForceWrapper):
    
    def __init__(self, floor_height, floor_axis, flip_floor=False):
        
        self.floor_height = floor_height 
        self.floor_axis = floor_axis
        self.flip_floor = flip_floor
        self.cached_hess = None
    
    def set_k(self, t):
        self.k = max(0, (t - 0.25) * 0.5)
    
    def _energy(self, x):
       
        #### Floor
        column_vec = x[:, self.floor_axis]
        row_vec = x[:, 1]
        target_height = self.k * row_vec + self.floor_height 
        # Calculate the distance of each y-coordinate from floor height
        distances = torch.abs(column_vec  - target_height)
        if(self.flip_floor):
            result = torch.where(column_vec <= target_height, torch.zeros_like(distances), distances)
        else:
            result = torch.where(column_vec >= target_height, torch.zeros_like(distances), distances)
        pt_wise_energy = result ** 2
        return pt_wise_energy

    def _gradient(self, x):
        return torch.autograd.functional.jacobian(lambda p: torch.sum(self.energy(p)), inputs=x)

    def _hessian(self, x):
        
        if self.cached_hess == None:
            self.cached_hess = torch.zeros(x.shape[0], x.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)

        self.cached_hess.zero_()
        column_vec = x[:, self.floor_axis]
        row_vec = x[:, 1]
        target_height = self.k * row_vec + self.floor_height 
        
        if (self.flip_floor):
            idx_below_floor = torch.nonzero(column_vec > target_height)
            self.cached_hess[idx_below_floor,idx_below_floor,self.floor_axis,self.floor_axis] = 2 
        else:
            idx_below_floor = torch.nonzero(column_vec < target_height)
            self.cached_hess[idx_below_floor,idx_below_floor,self.floor_axis,self.floor_axis] = 2 
        
        return self.cached_hess # big sparse tensor n x n x 3 x 3
    
# Spherical boundary condition
class Spheres(physics_utils.ForceWrapper):
    
    def __init__(self, center_1, center_2, radius_1, radius_2):
        
        self.center_1 = torch.tensor(center_1, device=device, dtype=torch.float32)
        self.center_2 = torch.tensor(center_2, device=device, dtype=torch.float32)
        self.radius_1 = radius_1
        self.radius_2 = radius_2
        self.cached_hess = None
    
    def _energy(self, x):
        
        distances_1 = torch.norm(x - self.center_1.unsqueeze(0), dim=1)
        distances_2 = torch.norm(x - self.center_2.unsqueeze(0), dim=1)
        result_1 = torch.where(distances_1 > self.radius_1, torch.zeros_like(distances_1), self.radius_1 - distances_1)
        result_2 = torch.where(distances_2 > self.radius_2, torch.zeros_like(distances_2), self.radius_2 - distances_2)
        pt_wise_energy = result_1 ** 2 + result_2 ** 2
        return pt_wise_energy

    def _gradient(self, x):
        return torch.autograd.functional.jacobian(lambda p: torch.sum(self.energy(p)), inputs=x)

    def _hessian(self, x):
        
        if self.cached_hess == None:
            self.cached_hess = torch.zeros(x.shape[0], x.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)

        self.cached_hess.zero_()
        distances_1 = torch.norm(x - self.center_1.unsqueeze(0), dim=1)
        distances_2 = torch.norm(x - self.center_2.unsqueeze(0), dim=1)
        idx_in_1 = torch.nonzero(distances_1 <= self.radius_1)
        idx_in_2 = torch.nonzero(distances_2 <= self.radius_2)
        idx_in = torch.cat([idx_in_1, idx_in_2], dim=0)
        self.cached_hess[idx_in,idx_in,0,0] = 2 
        self.cached_hess[idx_in,idx_in,1,1] = 2
        self.cached_hess[idx_in,idx_in,2,2] = 2 
        return self.cached_hess # big sparse tensor n x n x 3 x 3