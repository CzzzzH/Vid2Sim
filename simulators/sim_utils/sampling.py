import torch_geometric

def farthest_point_sampling(x, n_samples, random_start=False):
    x = x[0]
    ratio = n_samples / x.size(0) 
    idx = torch_geometric.nn.pool.fps(x, ratio=ratio, random_start=random_start)[:n_samples]
    return x[idx], idx 

