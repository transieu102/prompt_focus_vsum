import os
import cv2
import numpy as np
import torch
import clip
from tqdm import tqdm
from PIL import Image
import pickle
import gdown
# !pip install git+https://github.com/openai/CLIP.git
ids_dist  = {'P01.mp4': '1-0gKUHJIDlmVJnr9J7czvZMF-c1I6NYb',
            'P02.mp4': '1-DWZ3kR0al_Pr-HmxQwVhlrCJf2P7BbH',
            'P03.mp4': '1-ArbeGi0XLRzBVLCFMr8JhXveBY8gj4T',
            'P04.mp4': '1-0nBUN4YYin75_TCYHk8pkC6P9Bh08cr'}

video_dir = '/content/'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
for video in ids_dist.keys():
  video_path = os.path.join(video_dir, video)
  if not os.path.exists(video_path):
    url = f'https://drive.google.com/uc?id={ids_dist[video]}'
    gdown.download(url, video_path, quiet=False)



skip_frames = 5
save_dir = '/content/'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# clip.available_models()

# load model
model, preprocessing  = clip.load('RN50x64', device=device)

for video in os.listdir(video_dir):
  if not video.endswith('.mp4'):
    continue
  video_path = os.path.join(video_dir, video)
  cap = cv2.VideoCapture(video_path)
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  frames_feature = []
  frames_index_map = []
  frame_index = 0

  while frame_index < num_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = Image.fromarray(frame)
      frame = preprocessing(frame).unsqueeze(0).to(device)
      with torch.no_grad():
        feature = model.encode_image(frame)
      frames_feature.append(feature)
      frames_index_map.append(frame_index)
      frame_index += skip_frames
  with open(os.path.join(save_dir, "features_"+video.replace('mp4','pkl')), 'wb') as f:
    pickle.dump(frames_feature, f)
  with open(os.path.join(save_dir, "index_map_"+video.replace('mp4','pkl')), 'wb') as f:
    pickle.dump(frames_index_map, f)
  
    