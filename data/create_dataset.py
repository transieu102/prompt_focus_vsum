from data_tools import *
from utils import pre_video
# import clip
import torch
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
def create_new_features(clip_version : str, h5_file_name : str, h5_new_file_name : str, video_original_dir: str) -> None:
    """
    change features
    Args:
    h5_file_name : str
        h5 file name or save path
    model : object
        path to feature extractor
    h5_new_file_name : str
        h5 file name or save path
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, preprocessing  = clip.load(clip_version, device=device)
    data = read_h5_file(h5_file_name)
    
    for video_id in tqdm(data.keys()):
        cap = cv2.VideoCapture(os.path.join(video_original_dir, video_id+'.mp4'))
        features = []
        for i in data[video_id]['picks']:
            if not cap.isOpened():
                print("Error: Unable to open the video file.")
                return
            # Set the frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            # Read the frame
            ret, frame = cap.read()
            # Check if the frame is read successfully
            if not ret:
                print("Error: Unable to read frame.")
                return
            feature = extract_features(frame, model, preprocessing).reshape(-1)
            # print(feature.shape) 
            features.append(feature)
        # print(len(features))

        cap.release()
        # video_embeddings, video_mask = pre_video(torch.tensor(features), max_frames=512)
        video_embeddings = torch.tensor(features)
        video_mask = torch.ones(video_embeddings.size(0), dtype=torch.long)
        data[video_id]['video_embeddings'] = video_embeddings
        data[video_id]['video_mask'] = video_mask

        #TODO: add video caption features
        # print(features)
        prompt_embeddings = torch.zeros(features[0].shape[0])
        for frame_feature in features:
            prompt_embeddings += frame_feature
        prompt_embeddings /= len(features)
        data[video_id]['prompt_embedding'] = prompt_embeddings
    create_h5_file(data.keys(), data, h5_new_file_name)

def add_virtual_gtscores(h5_file_name : str, h5_new_file_name : str) -> None:
    #similarity
    data = read_h5_file(h5_file_name)
    for video_id in data.keys():
        concept_vector = data[video_id]['prompt_embedding']
        # print(concept_vector.shape)
        # print(data[video_id]['video_embeddings'][0].shape)
        similarity_scores = np.array([cosine_similarity(concept_vector.reshape(1, -1), data[video_id]['video_embeddings'][i].reshape(1, -1))[0][0] for i in range(len(data[video_id]['video_embeddings']))])
        data[video_id]['similarity_scores'] = similarity_scores
    
    #diversity
    alpha = 0.05
    for video_id in data.keys():
        diversity_scores = []
        n_frames = len(data[video_id]['video_embeddings'])
        scope = int(alpha*n_frames)
        for frame_id, frame_embedding in enumerate(data[video_id]['video_embeddings']):
            start = max(0, frame_id - scope)
            end = min(n_frames - 1, frame_id + scope)
            similarity_score = []
            for other_frame_id in range(start, end + 1):
                if other_frame_id != frame_id:
                    similarity_score.append(cosine_similarity(frame_embedding.reshape(1, -1), data[video_id]['video_embeddings'][other_frame_id].reshape(1, -1)))
            diversity_scores.append(-np.mean(similarity_score))
        data[video_id]['diversity_scores'] = diversity_scores
    create_h5_file(data.keys(), data, h5_new_file_name)
add_virtual_gtscores('dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14_useranno.h5', 'dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14_useranno_virtualgt.h5')

# data = read_h5_file('dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14_useranno_virtualgt.h5')
# for video_id in data.keys():
#     data[video_id]['similarity_scores'] = np.array([int(i) for i in data[video_id]['similarity_scores']])
#     # print(data[video_id]['similarity_scores'].shape)
#     # break
# create_h5_file(data.keys(), data, 'dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14_useranno_virtualgt.h5')