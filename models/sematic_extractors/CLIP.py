import torch 
import clip
from PIL import Image
import cv2
import numpy as np
class CLIP_Semantic_Extractor():
    def __init__(self, model="ViT-L/14", device = "cuda"):
        self.model, self.preprocess = clip.load(model, device=device)


    def extract_image_features(self, image, device = "cuda") -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(device)
        return self.model.encode_image(image).cpu().detach().numpy()
    

    def extract_query_features(self, query, device = "cuda") -> np.ndarray:
        return self.model.encode_text(clip.tokenize([query]).to(device)).cpu().detach().numpy()


    def extract_video_features(self, video, device = "cuda") -> np.ndarray:
        pass