
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention
)
from collections.abc import Iterable
from itertools import repeat

def _pair(v):
    if isinstance(v, Iterable):
        assert len(v) == 2, "len(v) != 2"
        return v
    return tuple(repeat(v, 2))

def infer_conv_output_dim(self, in_channels, input_dim):
    sample_seq_len = 200
    sample_bsz = 10

    # B: sample_bsz(batch_size), C: in_channels, S: sample_seq_len, D: input_dim
    x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
    # B x C x S x D

    for i, _ in enumerate(self.conv_layers):
        x = self.conv_layers[i](x)
    x = x.transpose(1, 2)
    # B x S x C x D

    mb, seq = x.size()[:2]

    # output_dimension = output_channels x output_feat_dim
    return x.contiguous().view(mb, seq, -1).size(-1)


class VGGBlock(nn.Module):
    """
        Input Tensor: (batch_size, in_channels=1, seq_len, inp_dim)
        Output Tensor: (batch_size, out_channels, seq_len, output_dim)
    """
    def __init__(
            self,
            inp_dim,
            in_channels,
            out_channels,
            conv_kernel_size,
            conv_stride,
            num_conv_layers,
            pooling_kernel_size,
            pooling_stride,
            padding=None,

    ):
        self.inp_dim = inp_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = _pair(conv_kernel_size)
        self.conv_stride = conv_stride
        self.pooling_kernel_size = _pair(pooling_kernel_size)
        self.pooling_stride = pooling_stride
        self.padding = (
            tuple(e // 2 for e in self.conv_kernel_size)
            if padding is None
            else _pair(padding)
        )

        self.layers = nn.ModuleList()
        for layer in range(num_conv_layers):
            conv_op = nn.Conv2d(
                in_channels=self.in_channels if layer == 0 else self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.conv_kernel_size,
                stride=self.conv_stride,
                padding=self.padding
            )
            self.layers.append(conv_op)
            self.layers.append(nn.ReLU())

        if self.pooling_kernel_size is not None:
            pool_op = nn.MaxPool2d(
                kernel_size=self.pooling_kernel_size,
                stride=self.pooling_stride,
                ceil_mode=True
            )
            self.layers.append(pool_op)
            self.total_output_dim, self.output_dim = infer_conv_output_dim(
                pool_op, inp_dim, self.out_channels
            )

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_embed_dim,
            dropout,
            num_heads,
            activation,
            normalize_before=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=self.dropout
        )

        self.layer_norm = LayerNorm(self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = normalize_before

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.activation_fn = utils.get_activation_fn(
            activation=activation
        )

    def forward(self, x):
        residual = x
        x = self.maybe_layer_norm(self.layer_norm, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x
        )
        x = F.dropout(x, p=self.dropout)
        x = self.maybe_layer_norm(self.layer_norm, x, after=True)
        x = residual + x

        residual = x
        x = self.maybe_layer_norm(self.layer_norm, x, before=True)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout)
        x = self.maybe_layer_norm(self.layer_norm, x, after=True)
        x = residual + x

        x = self.final_layer_norm(x)

        return x

    # MHA, FFN 이전에 LayerNorm을 적용할 지 결정하는 함수
    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        # 논문 상에선 normalize_before=True 로 구현합니다.
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class VGGTrf(nn.Module):
    """
        vggblock_config: ((out_channels, conv_kernel_size, conv_stride, num_conv_layers, pooling_kernel_size, pooling_stride, padding),) * num_vggblocks
            # 논문 상에서의 VGG layer parameter
            VGGBlock_1: out_channels=32, conv_kernel_size=3, conv_stride=1, num_conv_layers=2,
                    pooling_kernel_size=2, pooling_stride=2, padding=None
            VGGBlock_2: out_channels=64, conv_kernel_size=3, conv_stride=1, num_conv_layers=2,
                    pooling_kernel_size=2, pooling_stride=1, padding=None
            -> ((32, 3, 1, 2, 2, 2, None), (64, 3, 1, 2, 2, 1, None))

        transformer_config: ((embed_dim, num_heads, ffn_embed_dim, normalize_before, dropout),) * num_trfs
            # 논문 상에서의 Transformer layer parameter
            vggTrf(768, 12): embed_dim=768, num_heads=12, ffn_embed_dim=768*4, normalize_before=True, dropout=0.1
            -> ((768, 12, 768*4, True, 0.1),)*12
            vggTrf(512, 24)
            -> ((512, 24, 512*4, True, 0.1),)*24
    """
    def __init__(
            self,
            input_feat_dim,
            vggblock_config,
            transformer_config,
            output_dim # label 개수
    ):
        # VGG block의 개수
        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        ## VGG layer 구현 ##
        self.conv_layers = nn.ModuleList()
        self.input_dim = input_feat_dim
        self.in_channels = 1

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    conv_stride,
                    num_conv_layers,
                    pooling_kernel_size,
                    pooling_stride,
                    padding
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        inp_dim=input_feat_dim,
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        conv_kernel_size=conv_kernel_size,
                        conv_stride=conv_stride,
                        num_conv_layers=num_conv_layers,
                        pooling_kernel_size=pooling_kernel_size,
                        pooling_stride=pooling_stride,
                        padding=padding
                    )
                )
                self.in_channels = out_channels
                input_feat_dim = self.conv_layers[-1].ouput_dim


        ## Tranformer layer 구현 ##
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim
        )

        self.transformer_layers = nn.ModuleList()
        if transformer_input_dim != transformer_config[0][0]:
            # transformer sublayer의 embed dime과 input dim이 다를 경우 Linear 함수 적용
            self.transformer_layers.append(
                nn.Linear(transformer_input_dim, transformer_config[0][0])
            )
        self.transformer_layers.append(
            TransformerLayer(transformer_config[0])
        )

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(
                    nn.Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.transformer_layers.append(
                TransformerLayer(transformer_config[i])
            )

        ## Output layer (softmax) 구현 ##
        self.output_dim = output_dim
        self.output_layers = nn.ModuleList()
        self.output_layers.extend(
            [
                nn.Linear(transformer_config[-1][0], output_dim),
                nn.Softmax(output_dim)
            ]
        )

    def forward(self, x):
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        for i, _ in enumerate(self.transformer_layers):
            x = self.transformer_layers[i](x)
        for i, _ in enumerate(self.output_layers):
            x = self.output_layers[i](x)

        return x



