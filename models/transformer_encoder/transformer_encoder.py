from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        pass

    def forward(self, src, mask=None, src_key_padding_mask=None):
        pass