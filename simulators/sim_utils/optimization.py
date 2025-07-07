# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch 
from functools import partial

def line_search_0(func, x, direction, gradient, initial_step_size, alpha=1e-3, beta=0.6, max_steps=10):
    r"""Implements the simplest backtracking line search with the sufficient decrease Armijo condition

    Args:
        func (function): Optimization Energy/Loss function
        x (torch.Tensor): Optimization objective (the value you're optimizing)
        direction (torch.Tensor): Search direction
        gradient (torch.Tensor): Optimization gradient
        initial_step_size (float): Initial step size
        alpha (float, optional): LS parameter. Defaults to 1e-8.
        beta (float, optional): LS parameter. Defaults to 0.3.
        max_steps (int, optional): Max number of line search steps. Defaults to 20.

    Returns:
        torch float: Optimal step size used to scale the search direction
    """
    t = initial_step_size  # Initial step size
    f = func(x)
    gTd = gradient.T@direction

    for __ in range(max_steps):
        x_new = x + t * direction
        f_new = func(x_new)

        if f_new <= f + alpha * t * gTd:
            return t
        else:
            t *= beta  # Reduce the step size

    return t  # Return the final step size if max_iters is reached

def line_search_1(func, x, direction, gradient, initial_step_size, alpha=1e-3, beta=0.3, max_steps=10):
    r"""Implements a second version of line search where step size can also goes up once and then decrease until it satisfies the sufficient decrease Armijo condition

    Args:
        func (function): Optimization Energy/Loss function
        x (torch.Tensor): Optimization objective (the value you're optimizing)
        direction (torch.Tensor): Search direction
        gradient (torch.Tensor): Optimization gradient
        initial_step_size (float): Initial step size
        alpha (float, optional): LS parameter. Defaults to 1e-8.
        beta (float, optional): LS parameter. Defaults to 0.3.
        max_steps (int, optional): Max number of line search steps. Defaults to 20.

    Returns:
        torch float: Optimal step size used to scale the search direction
    """
    t = initial_step_size  # Initial step size
    f = func(x)
    gTd = gradient.T@direction
    can_break = False
    for __ in range(max_steps):
        x_new = x + t * direction
        f_new = func(x_new)
        if f_new <= f + alpha * t * gTd:
            if (can_break == True):
                return t
            else: 
                can_break = True
                t = t/beta  # Increase the step size
        else:
            t *= beta  # Reduce the step size

    return t  # Return the final step size if max_iters is reached

def newtons_method(x, energy_fcn, gradient_fcn, hessian_fcn, max_iters=10):
    r"""Newtons method

    Args:
        x (torch.Tensor): Variable, flattened vector, of shape :math:`(n,1)`
        energy_fcn (function): Calculates energy of the problem.
        gradient_fcn (function): Calculates the gradients
        hessian_fcn (function): Calculates the hessian
        max_iters (int, optional): Max newton iterations. Defaults to 10.
        conv_criteria (int, optional): Convergence criteria where 0: Gradient norm < eps, 1: step magnitude < eps. Defaults to 1.

    Returns:
        torch.Tensor: Optimized variable
    """
    for _ in range(max_iters):
        
        newton_G = gradient_fcn(x) 
        newton_H = hessian_fcn(x) 

        if torch.det(newton_H) == 0:
            newton_H = newton_H + 1e-4 * torch.eye(newton_H.shape[0], device=x.device)
        p = -torch.linalg.solve(newton_H, newton_G)
        if (torch.abs(newton_G.t() @ p) < 1e-3):
            break
        last_alpha = 10
        alpha = line_search_1(energy_fcn, x, p, newton_G, last_alpha, max_steps=10)
        x = x + alpha * p
    
    return x