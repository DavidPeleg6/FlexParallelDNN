import torch
import torch.nn as nn

dropout = 0.1
transformer_model = nn.Transformer(d_model=512, dim_feedforward=512, nhead=16, num_encoder_layers=12, batch_first=True,
                                   dropout=dropout, num_decoder_layers=0)
print([name for name, _ in transformer_model.named_modules()])
src = torch.rand((32, 128, 512))
# tgt = torch.rand((20, 32, 512))
tgt = src
# print(f'src: {src}, target: {tgt}')
out = transformer_model(src, tgt)
print(f'output: {out.shape}')

# 512 embedding (d_model)
# 512 hidden size (dim_feedforward)
# 128 seq length
# 12 encoder
# 16 heads
# 32 batch size
