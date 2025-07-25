import torch    
import json
import numpy as np
import random

from torch import nn
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor
from PIL import Image  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class SimulationDataset(Dataset):

    def __init__(self, dataset_dir, data_name=None, res=224, frame_num=16, split='test'):

        self.dataset_dir = dataset_dir
        self.frame_num = frame_num
        self.res = res 
        self.split = split

        self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False
        self.image_size = {"shortest_edge":res}
        
        if split in ['train', 'val']: # train/val with objaverse
            self.data = self.process_mesh_dataset(split)
        else: # test with GSO
            entry = {} 
            entry['frames'] = [f'{dataset_dir}/{data_name}/data/m_0_{i}.png' for i in range(frame_num)]  
            self.data = [entry]
    
    def process_mesh_dataset(self, split):
        
        data_list_path = f'{self.dataset_dir}/objaverse_{split}_list.json' 
        with open(data_list_path, 'r') as f:
            data_list = json.load(f) 

        data = []
        for data_info in data_list: 
            entry = {}
            entry['yms'] = data_info['yms']
            entry['prs'] = data_info['prs']
            obj_id = data_info['obj_id']
            entry['lbs'] = f'{self.dataset_dir}/outputs/{obj_id}/models/model_base.pth'
            entry['frames'] = [f'{self.dataset_dir}/outputs/{obj_id}/renderings/{i:03d}.png' for i in range(self.frame_num)] 
            data.append(entry)
        return data
    
    def to_white_background(self, img):
        rgba = img.convert('RGBA')
        img = np.ones([rgba.size[1], rgba.size[0], 3], dtype=np.uint8) * 255
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        try:
            entry = self.data[idx]
            video_raw = [ self.to_white_background(Image.open(frame)).resize((self.res, self.res)) for frame in entry['frames'] ]
            video = self.image_processor(video_raw, return_tensors="pt")['pixel_values'][0]
            
            if self.split in ['train', 'val']:  
                yms_gt = torch.log10(torch.tensor(entry['yms']).unsqueeze(0))
                prs_gt = torch.tensor(entry['prs']).unsqueeze(0)
                lbs_weight = torch.load(entry['lbs'], map_location='cpu')['net.18.weight']
                lbs_bias = torch.load(entry['lbs'], map_location='cpu')['net.18.bias']
                lbs_gt = torch.cat([lbs_weight.view(-1), lbs_bias.view(-1)])
                data = {'video': video, 'yms_gt': yms_gt, 'prs_gt': prs_gt, 'lbs_gt': lbs_gt} 
            else:
                data = {'video': video}

            return data

        except Exception as e:
            print(e)
            if self.split in ['train', 'val']:
                return self.__getitem__(random.randint(0, len(self.data) - 1))

class RegressionHead(nn.Module):

    def __init__(self, in_features=768):
        super(RegressionHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
class LBSHead(nn.Module):

    def __init__(self, in_features=768, hidden_dim=650, num_lbs_parameters=650):
        super(LBSHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_lbs_parameters)
        )
    
    def forward(self, x):
        return self.net(x)

class FeedForwardPredictor(nn.Module):

    def __init__(self, backbone, in_features=768):
        super(FeedForwardPredictor, self).__init__()

        self.backbone = backbone
        self.time_patch = backbone.config.num_frames // backbone.config.tubelet_size
        self.image_patch = backbone.config.image_size // backbone.config.patch_size
        self.regression_head_yms = RegressionHead(in_features)
        self.regression_head_prs = RegressionHead(in_features)
        self.lbs_head = LBSHead(in_features)
    
    def forward(self, x):

        x = self.backbone(pixel_values=x, return_dict=False)[0].mean(1)
        yms = self.regression_head_yms(x)
        prs = self.regression_head_prs(x)
        lbs = self.lbs_head(x)
        output = {'yms': yms, 'prs': prs, 'lbs': lbs}
        return output




    