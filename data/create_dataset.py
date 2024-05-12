from data_tools import *
from utils import pre_video
import clip
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
    
    for video_id in data.keys():
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
            features.append(extract_features(frame, model, preprocessing))
        cap.release()
        video_embeddings, video_mask = pre_video(torch.tensor(features), max_frames=512)
        data[video_id]['video_embeddings'] = video_embeddings
        data[video_id]['video_mask'] = video_mask

        #TODO: add video caption features
        prompt_embeddings = torch.zeros(features[0].size(1))
        for frame_feature in features:
            prompt_embeddings += frame_feature
        prompt_embeddings /= len(features)
        data[video_id]['prompt_embedding'] = prompt_embeddings
    create_h5_file(data.keys(), data, h5_new_file_name)