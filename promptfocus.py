import torch
from torch import nn
from models.temporal_transformer.temporal_transformer import create_tt
from torch.nn import MultiheadAttention
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    # TransformerDecoder,
    # TransformerDecoderLayer,
)

from models.transformer_decoder.transformer_decoder import LocalAttenModule


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
        self.tt, tt_width = create_tt(self.vit, depth=tt_depth)
        assert tt_width == self.vision_width

        # process prompt
        self.prompt_linear = nn.Linear(768, self.vision_width)

        # #query focus
        self.multihead_attention = MultiheadAttention(
            embed_dim=self.vision_width,
            num_heads=num_heads,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self.vision_width, nhead=num_heads
            ),
            num_layers=num_layers,
        )

        # TODO: replace by focal attetion
        self.transformer_decoder = LocalAttenModule(
            embed_dim=self.vision_width, kernel_size=kernel_size
        )
        self.kernel_size = kernel_size

        self.linear = nn.Linear(self.vision_width, 1)
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
        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(
            self.position_embeddings.weight, video_embeddings.size(1)
        )
        # position_embeddings = position_embeddings
        video_embeddings = video_embeddings + position_embeddings
        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask)  # shape dont change

        video_embeddings = video_embeddings + position_embeddings
        # multihead attention
        # TODO: gen caption from video & input to attention heads
        # prompt_embeddings = self.prompt_linear(prompt_embeddings)
        video_embeddings = video_embeddings.permute(1, 0, 2)
        video_embeddings_attn, attn_weights = self.multihead_attention(
            query=video_embeddings, key=prompt_embeddings, value=prompt_embeddings
        )

        # #transformer scoring
        video_embeddings_enc = self.transformer_encoder(video_embeddings_attn)

        video_embeddings_enc = video_embeddings_enc.permute(1, 0, 2)

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
            video_embeddings_enc, attention_mask[:, None, :, :]
        )
        score = self.linear(video_embeddings_dec)
        return score, video_embeddings_dec.permute(1, 0, 2)


if __name__ == "__main__":
    model = PromptFocus()
    model.to("cuda")
    # RANDOM video embeddings, video mask, prompt embeddings
    video_embeddings = torch.randn(1294,1, 768).to("cuda")
    video_mask = torch.randn(1, 1294).to("cuda")
    prompt_embeddings = torch.randn(1, 1, 768).to("cuda")
    score = model(video_embeddings, video_mask, prompt_embeddings)
    print(score.shape)
