import torch
import torch.nn as nn
from transformerlayers.sublayers3 import MultiHeadAttention,MultiHeadAttention1,MultiHeadAttention2

from transformerlayers.sublayers3 import PoswiseFeedForwardNet


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention1(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs1=self.norm1(enc_inputs+enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs1)
        enc_outputs=self.norm2(enc_outputs1+enc_outputs)

        return enc_outputs, attn

class DecoderLayer2(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer2, self).__init__()
        self.dec_self_attn = MultiHeadAttention2(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        # 双向注意力层
        dec_outputs, dec_self_attn, dec_cross_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask)
        dec_outputs1 = self.norm1(dec_inputs + dec_outputs)

        # 位置前馈网络
        dec_outputs = self.pos_ffn(dec_outputs1)
        dec_outputs = self.norm3(dec_outputs1 + dec_outputs)

        return dec_outputs, dec_self_attn, dec_cross_attn
class DecoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, attn_mask=self_attn_mask)
        dec_outputs1=self.norm1(dec_inputs+dec_outputs)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs1, enc_outputs,
                                                      enc_outputs, attn_mask=enc_attn_mask)
        dec_outputs2=self.norm2(dec_outputs1+dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs2)
        dec_outputs=self.norm3(dec_outputs2+dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class EncoderLayer1(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer1, self).__init__()

        self.self_local = MultiHeadAttention1(d_model, d_k, d_v, n_heads, dropout)
        self.self_global = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        # self.global_grid = MultiHeadAttention1(d_model, d_k, d_v, h, dropout)
        # self.global_region = MultiHeadAttention1(d_model, d_k, d_v, h, dropout)

        self.cls_local = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
        self.cls_global = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        self.pwff_local = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.pwff_global = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        b_s = enc_inputs.shape[0]
        cls_local = self.cls_local.expand(b_s, 1, -1)
        cls_global = self.cls_global.expand(b_s, 1, -1)

        # cls_local = self.global_local(cls_local, enc_inputs, enc_inputs)
        # cls_global = self.global_local(cls_global, enc_inputs, enc_inputs, attention_mask=self_attn_mask)

        local_features = torch.cat([cls_local, enc_inputs], dim=1)
        global_features = torch.cat([cls_global, enc_inputs], dim=1)

        # add_mask = torch.zeros(b_s, 142, 1).bool().to(global_features.device)
        # self_attn_mask = torch.cat([add_mask, enc_inputs], dim=-1)
        local_att = self.self_local(local_features, local_features, local_features,attn_mask = self_attn_mask)
        global_att = self.self_global(global_features, global_features, global_features, attn_mask = self_attn_mask)

        local_ff = self.pwff_grid(local_att)
        global_ff = self.pwff_region(global_att)

        enc_outputs,att = torch.cat(local_ff,global_ff)


        # local_ff = local_ff[:, 1:]
        # global_ff = global_ff[:, 1:]

        return enc_outputs,att
