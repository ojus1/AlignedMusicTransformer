from custom.config import config

import sys
sys.path.append("RWKV-v4neo/")
from src.model import RWKV
import torch
import torch.distributions as dist
import random
import utils
from tqdm import tqdm
import os
os.environ["RWKV_FLOAT_MODE"] = "fp16"
import torch

class Args:
    load_model = ""
    wandb = ""
    proj_dir = "out"
    random_seed = -1
    data_file = ""
    data_type = "utf-8"
    vocab_size = 388+2
    ctx_len = 2048
    epoch_steps = 1000
    epoch_count = 500
    epoch_begin = 0
    epoch_save = 5
    micro_bsz = 12
    n_layer = 6
    n_embd = 256
    pre_ffn = 0
    head_qk = 0
    tiny_att_dim = 0
    tiny_att_layer = -999
    lr_init = 6e-4
    lr_final = 1e-5
    warmup_steps = 0
    beta1 = 0.9
    beta2 = 0.99
    adam_eps = 1e-8
    grad_cp = 0
    my_pile_stage = 0
    my_pile_shift = -1
    my_pile_edecay = 0
    layerwise_lr = 1
    ds_bucket_mb = 200
    my_img_version = 0
    my_img_size = 0
    my_img_bit = 0
    my_img_clip = 'x'
    my_img_clip_scale = 1
    my_img_l1_scale = 0
    my_img_encoder = 'x'
    my_sample_len = 0
    my_ffn_shift = 1
    my_att_shift = 1
    my_pos_emb = 0
    load_partial = 0
    magic_prime = 0
    my_testing = 0
    
args = Args()

class MusicTransformerRWKV(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        args.n_embd = embedding_dim
        args.vocab_size = vocab_size
        args.n_layer = num_layer
        args.ctx_len = max_seq
        
        self.decoder = RWKV(args)

    def forward(self, x, length=None, writer=None):
        if self.training or not self.infer:
            # _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.pad_token)
            decoder = self.decoder(x)
            # print(decoder.shape, "inside forward")
            # return fc.contiguous() if self.training else (fc.contiguous(), [weight.contiguous() for weight in w])
            return decoder
        else:
            return self.generate(x, length, None).contiguous().tolist()

    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 tf_board_writer=None):
        decode_array = prior
        result_array = prior
        # print(config)
        # print(length)
        for i in tqdm(range(length)):
            if decode_array.size(1) >= config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            result = self.decoder(decode_array)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = 0
            if u > 1:
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        result_array = result_array[0]
        return result_array

    def test(self):
        self.eval()
        self.infer = True
