import torch
from inspect import isfunction
from torch import nn, einsum
from models.temporal_transformer.temporal_transformer import create_tt
from torch.nn import MultiheadAttention
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from einops import rearrange, repeat
from sklearn.metrics.pairwise import cosine_similarity
from models.transformer_decoder.transformer_decoder import LocalAttenModule

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        # if _ATTN_PRECISION =="fp32":
        with torch.autocast(enabled=False, device_type = 'cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # else:
            # sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class PromptFocus(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        tt_depth: int = 1,
        num_layers: int = 6,
        kernel_size=5,
        loss_type="cross_entropy",
        vit="base",
        max_length=1294, #max length of video
    ):
        super(PromptFocus, self).__init__()
        self.max_length = max_length
        self.vit = vit
        assert vit in ["base", "large"]
        if vit == "base":
            self.vision_width = 768
        elif vit == "large":
            self.vision_width = 1024
        # create temporal transformer
        self.position_embeddings = nn.Embedding(max_length, self.vision_width)
        self.tt, tt_width = create_tt(self.vit,num_heads=16, depth=tt_depth)
        assert tt_width == self.vision_width

        # process prompt
        self.prompt_linear = nn.Linear(768, self.vision_width)

        # #query focus
        self.multihead_attention = MultiheadAttention(
            embed_dim=self.vision_width,
            num_heads=num_heads,
        )

        # TODO: replace by focal attetion
        self.transformer_decoder = LocalAttenModule(
            embed_dim=self.vision_width, kernel_size=kernel_size
        )
        self.kernel_size = kernel_size

        self.linear = nn.Linear(self.vision_width, 1)

        self.cross_attention = CrossAttention(
            query_dim=self.vision_width,
            context_dim=self.vision_width,
            heads=16,
            dim_head=self.vision_width,
            dropout=0.3,
        )

        self.reconstruct_decoder = nn.Linear(self.vision_width, self.vision_width)
        # self.sigmod = nn.Sigmoid()
    def _interpolate_pos_embed(self, pos_embed, video_length):
        if video_length > self.max_length:
            pos_embed = torch.nn.functional.interpolate(
                pos_embed[:, None, None, :].permute(1, 3, 0, 2),
                size=(video_length, 1),
                mode="bicubic",
                align_corners=False,
            )[0, :, :, 0].permute(1, 0)
        else:
            pos_embed = pos_embed[:video_length]

        return pos_embed

    def forward(self, video_embeddings, video_mask, prompt_embeddings):
        """

        Args:
            video_embeddings (torch.Tensor): video embeddings (B, T, C)
            video_mask (torch.Tensor): video mask (B, T)
            prompt_embeddings (torch.Tensor): prompt embeddings (B, C)
        """
        device = video_embeddings.device
        video_len = video_embeddings.size(0)

        video_embeddings = video_embeddings.permute(1, 0, 2)

        video_embeddings_attn = self.cross_attention(
            video_embeddings, prompt_embeddings
        )

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(
            self.position_embeddings.weight, video_embeddings_attn.size(1)
        )
        # position_embeddings = position_embeddings
        video_embeddings_attn = video_embeddings_attn + position_embeddings
        # temporal transformer
        video_embeddings_tt = self.tt(video_embeddings_attn, video_mask)  # shape dont change
        

        # similarity weight
        # for idx in range(video_embeddings.shape[1]):
        #     feature = video_embeddings[:, idx, :]
        #     similarity_score = cosine_similarity(
        #         feature.detach().cpu().numpy(), 
        #         prompt_embeddings.squeeze(0).detach().cpu().numpy()
        #     )[0][0]
        #     feature = feature * similarity_score

        # multihead attention
        # TODO: gen caption from video & input to attention heads
        # prompt_embeddings = self.prompt_linear(prompt_embeddings)
        # video_embeddings = video_embeddings.permute(1, 0, 2)
        # video_embeddings_attn = self.cross_attention(
        #     video_embeddings, prompt_embeddings
        # )
        # video_embeddings_attn, attn_weights = self.multihead_attention(
        #     query=video_embeddings, key=prompt_embeddings, value=prompt_embeddings
        # )
        # #transformer scoring
        # video_embeddings_attn = video_embeddings_attn.permute(1, 0, 2)
        # video_embeddings_enc = self.transformer_encoder(video_embeddings_attn)
        # video_embeddings_enc = self.transformer_encoder(video_embeddings)
        
        attention_mask = torch.zeros([video_len, video_len])
        half_atten_len = min(self.kernel_size // 2 + 1, video_len)
        for j in range(half_atten_len):
            attention_mask += torch.diag(torch.ones(video_len - j), diagonal=j)
            if j > 0:
                attention_mask += torch.diag(torch.ones(video_len - j), diagonal=-j)
        attention_mask = attention_mask.to(device)
        attention_mask = (
            attention_mask[None, :, :] * video_mask[:, None, :] * video_mask[:, :, None]
        )
        video_embeddings_dec = self.transformer_decoder(
            video_embeddings_tt, attention_mask[:, None, :, :]
        )
        score = self.linear(video_embeddings_dec)

        #TODO: transformer decode -> reconstruct
        video_embeddings_scored = video_embeddings * score
        video_embeddings_reconstructed = self.reconstruct_decoder(
            video_embeddings_scored
        )

        return score, video_embeddings_dec.permute(1, 0, 2), video_embeddings_reconstructed.permute(1, 0, 2)


# if __name__ == "__main__":
#     model = PromptFocus()
#     model.to("cuda")
#     # RANDOM video embeddings, video mask, prompt embeddings
#     video_embeddings = torch.randn(1294,1, 768).to("cuda")
#     video_mask = torch.randn(1, 1294).to("cuda")
#     prompt_embeddings = torch.randn(1, 1, 768).to("cuda")
#     score = model(video_embeddings, video_mask, prompt_embeddings)
#     print(score.shape)
