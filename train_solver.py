from data.data_tools import read_h5_file
from promptfocus import PromptFocus
import yaml
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from solver_utils import (
    cosine_lr_schedule,
    generate_summary,
    represent_features

)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Solver(object):
    def __init__(self):
        self.model = None
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    def initualize(self):
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = "cpu"
        # fix the seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True
<<<<<<< HEAD
        self.model = PromptFocus(self.config['num_heads'], self.config['tt_depth'], self.config['num_layers'], self.config['kernel_size'], self.config['loss_type'], self.config['vit'], 
        max_length=self.config['max_video_length']).to(self.device)
=======
        self.model = PromptFocus(self.config['num_heads'], self.config['tt_depth'], self.config['num_layers'], self.config['kernel_size'], self.config['loss_type'], self.config['vit'], max_length = self.config['max_video_length']).to(self.device)
>>>>>>> f6641521ce579c9da8c887f21651d60fd5897f21
    def load_dataset(self):
        self.dataset = read_h5_file(self.config['dataset_path'])
    def load_split(self):
        with open (self.config['split_path'], 'r') as f:
            self.split = json.load(f)

    def representativeness_loss(self, outputs, targets):
        criterion = nn.MSELoss()
        return criterion(outputs, targets)

    def diversity_loss(self, features):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        # Compute diversity loss (negative sum of similarities)
        diversity_loss = -torch.sum(similarity_matrix) / (features.size(0) * (features.size(0) - 1))
        return diversity_loss

    def train(self, split_ids: list = None):
        print("Load dataset...")
        self.load_dataset()
        print("Load split...")
        self.load_split()
        if split_ids is None:
            split_ids = [i for i in range(len(self.split))]
        for split_id in split_ids:
            print('Begin training on split {}'.format(split_id))
            train_keys = self.split[split_id]['train_keys']
            model_vram = torch.cuda.memory_allocated()
            self.initualize()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config['init_lr']))
            for epoch in range(self.config['max_epoch']):
                cosine_lr_schedule(
                optimizer,
                epoch,
                int(self.config["max_epoch"]),
                float(self.config["init_lr"]),
                float(self.config["min_lr"]),
                )
                self.model.train()
                # for video_id in tqdm(train_keys):
                for video_id in train_keys:
                    video_embeddings = torch.tensor(self.dataset[video_id]['video_embeddings']).to(self.device).unsqueeze(1)
                    video_mask = torch.tensor(self.dataset[video_id]['video_mask']).to(self.device).unsqueeze(0)
<<<<<<< HEAD
                    prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding']).to(self.device).unsqueeze(0).unsqueeze(0)
                    
                    # print('video feature shape:', video_embeddings.shape)
                    # print(' video_mask shape:', video_mask.shape)
                    # print('prompt_embeddings shape:', prompt_embeddings.shape)
=======
                    prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding']).to(self.device)
                    print('video feature shape:', video_embeddings.shape)
                    print(' video_mask shape:', video_mask.shape)
                    print('prompt_embeddings shape:', prompt_embeddings.shape)
>>>>>>> f6641521ce579c9da8c887f21651d60fd5897f21
                    score = self.model(video_embeddings, video_mask, prompt_embeddings)
                    score = score.detach().cpu().numpy().squeeze(0).squeeze(1)
                    print("score",score.shape)
                    summary,_ = generate_summary(score, self.dataset[video_id]['change_points'], self.dataset[video_id]['n_frames'], self.dataset[video_id]['n_frame_per_seg'], self.dataset[video_id]['picks'])
                    print(np.array(summary).shape)
                    mask = [gt for frame_id, gt in enumerate(summary) if frame_id in self.dataset[video_id]['picks'] ]
                    represented_features = represent_features(mask, video_embeddings)
                    representativeness_loss = self.representativeness_loss(represented_features, video_embeddings)
                    diversity_loss = self.diversity_loss(video_embeddings[mask == 1])
                    # loss = representativeness_loss + diversity_loss
                    loss =  diversity_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



solver = Solver()
solver.load_config('/MLCV/haov/projects/video-sum/prompt_focus_vsum/config/promt_focus.yaml')
solver.train()