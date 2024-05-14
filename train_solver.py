from data.data_tools import read_h5_file
from promptfocus import PromptFocus
import yaml
import os
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
    represent_features,
    evaluate_summary,
    calculate_rank_order_statistics


)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Solver(object):
    def __init__(self):
        self.model = None
        self.evaluate_result = {}
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    def initualize(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)
        # self.device = "cpu"
        # fix the seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True
        self.model = PromptFocus(self.config['num_heads'], self.config['tt_depth'], self.config['num_layers'], self.config['kernel_size'], self.config['loss_type'], self.config['vit'], 
        max_length=self.config['max_video_length']).to(self.device)
    def load_dataset(self):
        self.dataset = read_h5_file(self.config['dataset_path'])
    def load_split(self):
        with open (self.config['split_path'], 'r') as f:
            self.split = json.load(f)

    def representativeness_loss(self, outputs, targets):
        # print(outputs, targets)
        criterion = nn.MSELoss()
        return criterion(outputs, targets)

    def diversity_loss(self, features):
        # print(features[:5])
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        # Compute diversity loss (negative sum of similarities)
        diversity_loss = torch.sum(similarity_matrix) / (features.size(0) * (features.size(0) - 1))
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
            criterion = nn.MSELoss()

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
                for video_id in tqdm(train_keys):
                    video_embeddings = torch.tensor(self.dataset[video_id]['video_embeddings'], dtype=torch.float32).to(self.device).unsqueeze(1)
                    video_mask = torch.tensor(self.dataset[video_id]['video_mask'], dtype=torch.float32).to(self.device).unsqueeze(0)
                    prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding']).to(self.device).unsqueeze(0).unsqueeze(0)
                    
                    # print('video feature shape:', video_embeddings.shape)
                    # print(' video_mask shape:', video_mask.shape)
                    # print('prompt_embeddings shape:', prompt_embeddings.shape)
                    score, video_embeddings_dec = self.model(video_embeddings, video_mask, prompt_embeddings)
                    # score_np = score.detach().cpu().numpy().squeeze(0).squeeze(1)
                    # summary,_ = generate_summary(
                    #     score_np, 
                    #     self.dataset[video_id]['change_points'], 
                    #     self.dataset[video_id]['n_frames'], 
                    #     self.dataset[video_id]['n_frame_per_seg'], 
                    #     self.dataset[video_id]['picks']
                    #     )
                    # mask = [gt for frame_id, gt in enumerate(summary) if frame_id in self.dataset[video_id]['picks'] ]
                    # represented_features = represent_features(
                    #     mask, 
                    #     video_embeddings_dec, 
                    #     [z for z in self.dataset[video_id]['change_points']], 
                    #     [z for z in self.dataset[video_id]['picks']],
                    #     device=self.device
                    # )
                    # representativeness_loss = self.representativeness_loss(represented_features, video_embeddings)
                    # diversity_loss = self.diversity_loss(video_embeddings_dec[np.array(mask) == 1])

                    label = torch.tensor(self.dataset[video_id]['similarity_scores'], dtype=torch.float32).to(self.device)
                    label = label.unsqueeze(0).unsqueeze(2)
                    print(score.shape)
                    print(label.shape)
                    # print(label)

                    loss = criterion(score, label) 
                    # print
                    # loss = representativeness_loss 
                    # loss.requires_grad = True
                    optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    optimizer.step()
                if epoch % 10 == 0:
                  print('Epoch:', epoch)
                  print('Loss:', loss)
                  self.evaluate(split_id, epoch)
                  input()
    def evaluate(self, split_id, epoch: int):
            self.model.eval()
            test_keys = self.split[split_id]['test_keys']
            split_f1 = []
            split_kscore = []
            split_spearn = []
            # plot = 1
            for video_id in test_keys:
                video_embeddings = torch.tensor(self.dataset[video_id]['video_embeddings'], dtype=torch.float32).to(self.device).unsqueeze(1)
                video_mask = torch.tensor(self.dataset[video_id]['video_mask'], dtype=torch.float32).to(self.device).unsqueeze(0)
                prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding']).to(self.device).unsqueeze(0).unsqueeze(0)
                score, _ = self.model(video_embeddings, video_mask, prompt_embeddings)
                score = score.detach().cpu().numpy().squeeze(0).squeeze(1)
                # print(score)
                # print(self.dataset[video_id].keys())
                summary, framescores = generate_summary(
                            score, 
                            self.dataset[video_id]['change_points'], 
                            self.dataset[video_id]['n_frames'], 
                            self.dataset[video_id]['n_frame_per_seg'], 
                            self.dataset[video_id]['picks']
                            )
                # if plot:
                #     import matplotlib.pyplot as plt
                #     plt.plot(score)
                #     plt.plot(framescores)
                #     plt.show()
                #     plot = 0
                final_f_score, final_prec, final_rec = evaluate_summary(
                    summary, 
                    self.dataset[video_id]['user_summary'],
                    
                    )
                split_f1.append(final_f_score)
                # user_anno = get_user_anno_tvsum(annot=annot, video_name=video_id)
                user_anno =self.dataset[video_id]['user_anno'][()].T
    
                # tính kendalltau score
                kscore = calculate_rank_order_statistics(frame_scores=framescores,user_anno=user_anno, metric="kendalltau")
                # kscores.append(kscore)
                split_kscore.append(kscore)


                # tính kendalltau score
                sscore = calculate_rank_order_statistics(frame_scores=framescores, user_anno=user_anno, metric="spearman")
                # sscores.append(sscore)
                split_spearn.append(sscore)
            print("Split ", split_id)        
            print('F-score:', np.mean(final_f_score))
            print('kendalltau score:', np.mean(kscore))
            print('spearman score:', np.mean(sscore))
            self.evaluate_result[split_id] = [np.mean(final_f_score), np.mean(kscore), np.mean(sscore)]
            
            os.makedirs(self.config['save_dir'], exist_ok=True)
            self.save_model(self.config['save_dir'] + '/' + f"model_tvsum_{split_id}_{np.mean(final_f_score)}_{np.mean(kscore)}_{np.mean(sscore)}_{epoch}" + '.pt')

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        return torch.load(path)
solver = Solver()
solver.load_config('config/promt_focus.yaml')
solver.train()