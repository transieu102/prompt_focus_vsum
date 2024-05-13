import torch
def pre_video(video_embeddings, max_frames=512):
    video_mask = torch.ones(video_embeddings.size(0), dtype=torch.long)

    if video_embeddings.size(0) > max_frames:
        video_embeddings = video_embeddings[:max_frames]
        video_mask = video_mask[:max_frames]
    return video_embeddings, video_mask

