import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import torch  # noqa

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
