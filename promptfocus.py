import torch
from torch import nn
from models.temporal_transformer.temporal_transformer import create_tt
from models.multihead_attention.multihead_attention import MultiHeadAttention
from models.transformer_encoder.transformer_encoder import TransformerEncoder
from models.transformer_decoder.transformer_decoder import TransformerDecoder
class PromptFocus(nn.Module):
    def __init__(self, 
                 tt_depth=1,
                 loss_type='cross_entropy',
                 vit='base'):
        super(PromptFocus, self).__init__()
        self.vit = vit
        assert vit in ['base', 'large']
        if vit == 'base':
            self.vision_width = 768
        elif vit == 'large':
            self.vision_width = 1024
         # create temporal transformer
        self.position_embeddings = nn.Embedding(512, self.vision_width)
        self.tt, tt_width = create_tt(
            self.vit, depth=tt_depth)
        assert tt_width == self.vision_width

        #process prompt
        self.prompt_linear = nn.Linear(768, self.vision_width)

        #query focus
        self.multihead_attention = MultiHeadAttention()

        self.transformer_encoder = TransformerEncoder()

        self.transformer_decoder = TransformerDecoder()

    def _interpolate_pos_embed(self, pos_embed, video_length):
        if video_length > 512:
            pos_embed = torch.nn.functional.interpolate(
                pos_embed[:, None, None, :].permute(1, 3, 0, 2),
                size=(video_length, 1), mode='bicubic', align_corners=False)[0, :, :, 0].permute(1, 0)
        else:
            pos_embed = pos_embed[:video_length]
        return pos_embed

    def forward(self, video_embeddings, video_mask, prompt_embeddings):
        #video_embedding [numframe, embed_width]
        device = video_embeddings.device 
        
        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1)) # size 1 ???
        video_embeddings = video_embeddings + position_embeddings


        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask) # shape dont change
        video_embeddings = video_embeddings + position_embeddings 

        #multihead attention
        #TODO: gen caption from video & input to attention heads
        prompt_embeddings = self.prompt_linear(prompt_embeddings)
        
        video_embeddings = self.multihead_attention(q = video_embeddings, k = prompt_embeddings, v = prompt_embeddings)

        #transformer scoring
        video_embeddings = self.transformer_encoder(video_embeddings, video_mask)
        score = self.transformer_decoder(video_embeddings, video_mask)
        return score