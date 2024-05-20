from data.data_tools import read_h5_file
from promptfocus import PromptFocus
import yaml
import os
import csv
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from solver_utils import (
    cosine_lr_schedule,
    generate_summary,
    represent_features,
    evaluate_summary,
    calculate_rank_order_statistics


)

def get_user_anno_tvsum(
    annot: list,
    video_name: str
) -> list :
    user = int(video_name.split("_")[-1])

    annotation_length = list(Counter(np.array(annot)[:, 0]).values())[user-1]
    init = (user - 1) * annotation_length
    till = user * annotation_length

    user_scores = []
    for row in annot[init:till]:
        curr_user_score = row[2].split(",")
        curr_user_score = np.array([float(num) for num in curr_user_score])
        curr_user_score = curr_user_score / curr_user_score.max(initial=-1)  # Normalize scores between 0 and 1
        # curr_user_score = curr_user_score[::15] # REMOVE-ME

        user_scores.append(curr_user_score)
    return user_scores

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Solver(object):
    def __init__(self):
        self.best_models = {}
        self.best_results = {}
        self.evaluate_result = {}
        self.model = None

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
        # for i in range(len(self.split)):
        #     self.best_models[i] = PromptFocus(self.config['num_heads'], self.config['tt_depth'], self.config['num_layers'], self.config['kernel_size'], self.config['loss_type'], self.config['vit'], 
        #     max_length=self.config['max_video_length']).to(self.device)

    def load_dataset(self):
        self.dataset = read_h5_file(self.config['dataset_path'])
    def load_split(self):
        with open (self.config['split_path'], 'r') as f:
            self.split = json.load(f)
        for i in range(len(self.split)):
            self.best_models[i] = PromptFocus(self.config['num_heads'], self.config['tt_depth'], self.config['num_layers'], self.config['kernel_size'], self.config['loss_type'], self.config['vit'], 
            max_length=self.config['max_video_length']).to(self.device)
            self.best_results[i] = 0
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
        # self.initualize()
        # self.evaluate_all_split()
        if split_ids is None:
            split_ids = [i for i in range(len(self.split))]
        for split_id in split_ids:
            self.initualize()
            print('Begin training on split {}'.format(split_id))
            train_keys = self.split[split_id]['train_keys']
            # model_vram = torch.cuda.memory_allocated()
            # self.evaluate(split_id, -1)
            # input()
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
                    prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding'], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                    # print("video_embeddings",video_embeddings.shape)
                    # print("video_mask",video_mask.shape)
                    # print("prompt_embeddings",prompt_embeddings.shape)

                    # print('video feature shape:', video_embeddings.shape)
                    # print(' video_mask shape:', video_mask.shape)
                    # print('prompt_embeddings shape:', prompt_embeddings.shape)
                    score, video_embeddings_dec , video_embeddings_reconstructed = self.model(video_embeddings, video_mask, prompt_embeddings)
                    score_np = score.detach().cpu().numpy().squeeze(0).squeeze(1)
                    summary,_ = generate_summary(
                        score_np, 
                        self.dataset[video_id]['change_points'], 
                        self.dataset[video_id]['n_frames'], 
                        self.dataset[video_id]['n_frame_per_seg'], 
                        self.dataset[video_id]['picks']
                        )
                    mask = [gt for frame_id, gt in enumerate(summary) if frame_id in self.dataset[video_id]['picks'] ]
                    
                    represented_video_embeddings_reconstructed = represent_features(
                        mask, 
                        video_embeddings_reconstructed, 
                    )   
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
                    reconstruct_loss =  criterion(video_embeddings_reconstructed, video_embeddings) # NOTE: ~ 0.4
                    
                    prompt_embeddings = prompt_embeddings.repeat(video_embeddings.shape[0], 1, 1)
                    represented_prompt_embeddings = represent_features(
                        mask, 
                        prompt_embeddings, 
                    )  
                    prompt_loss = criterion(
                        represented_video_embeddings_reconstructed,
                        represented_prompt_embeddings
                    ) # NOTE: ~ 0.05
                    
                    similarity_loss = criterion(score, label) # NOTE: ~0.2

                    loss = reconstruct_loss + prompt_loss * 10 + similarity_loss * 2

                    optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Epoch:', epoch)
                    print('Loss:', loss)
                    self.evaluate(split_id, epoch, 'avg')
        self.evaluate_all_split('avg')
                #   input()
    def evaluate(self, split_id, epoch: int, eval_metric='avg'):
            test_keys = self.split[split_id]['test_keys']
            split_f1 = []
            if eval_metric == 'avg':
                split_kscore = []
                split_spearn = []
            # plot = 1
            for video_id in test_keys:
                video_embeddings = torch.tensor(self.dataset[video_id]['video_embeddings'], dtype=torch.float32).to(self.device).unsqueeze(1)
                video_mask = torch.tensor(self.dataset[video_id]['video_mask'], dtype=torch.float32).to(self.device).unsqueeze(0)
                prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding'] , dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                score, _, video_embeddings_reconstructed = self.model(video_embeddings, video_mask, prompt_embeddings)
                # print(_)
                
                score = score.detach().cpu().numpy().squeeze(0).squeeze(1)
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
                    eval_metric
                    )
                split_f1.append(final_f_score)
                if eval_metric == 'avg':
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
            print('F-score:', np.mean(split_f1))
            if eval_metric == 'avg':
                print('kendalltau score:', np.mean(split_kscore))
                print('spearman score:', np.mean(split_spearn))
            if np.mean(split_f1) > self.best_results[split_id]:
                self.best_results[split_id] = np.mean(split_f1)
                self.best_models[split_id].load_state_dict(self.model.state_dict())
           
    def evaluate_all_split(self, eval_metric='avg'):
        f1 = []
        if eval_metric == 'avg':
            ks = []
            sp = []
        for split_id in range(len(self.split)):
            test_keys = self.split[split_id]['test_keys']
            split_f1 = []
            if eval_metric == 'avg':
                split_kscore = []
                split_spearn = []
            # plot = 1
            for video_id in test_keys:
                video_embeddings = torch.tensor(self.dataset[video_id]['video_embeddings'], dtype=torch.float32).to(self.device).unsqueeze(1)
                video_mask = torch.tensor(self.dataset[video_id]['video_mask'], dtype=torch.float32).to(self.device).unsqueeze(0)
                prompt_embeddings = torch.tensor(self.dataset[video_id]['prompt_embedding'], dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
                score, _, video_embeddings_reconstructed = self.best_models[split_id](video_embeddings, video_mask, prompt_embeddings)
                # print(_)
                
                score = score.detach().cpu().numpy().squeeze(0).squeeze(1)
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
                    eval_metric
                    )
                split_f1.append(final_f_score)

                if eval_metric == 'avg':
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
            print('F-score:', np.mean(split_f1))
            f1.append(np.mean(split_f1))
            if eval_metric == 'avg':
                print('kendalltau score:', np.mean(split_kscore))
                print('spearman score:', np.mean(split_spearn))
                ks.append(np.mean(split_kscore))
                sp.append(np.mean(split_spearn))
        print("Average:")
        print('F-score:', np.mean(f1))
        if eval_metric == 'avg':
            print('kendalltau score:', np.mean(ks))
            print('spearman score:', np.mean(sp))
solver = Solver()
solver.load_config('config/promt_focus.yaml')
solver.train()