import torch

import torch.nn.functional as F
import torchvision.models as models
from torch import nn, einsum, Tensor
from functools import partial
import transformers
import matplotlib
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, pack, unpack, repeat, reduce


from open_clip import create_model_from_pretrained, get_tokenizer  # works on open-clip-torch>=2.23.0, timm>=0.9.8


# Clip的文本编码器，获取文本编码
class BiomedCLIP(torch.nn.Module):
    def __init__(self):
        super(BiomedCLIP, self).__init__()
        self.model, self.processor = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.model.eval()
        self.context_length = 256

    def forward(self, texts):
        batch_size = texts.shape[0]
        image = torch.zeros((batch_size, 3, 224, 224)).to(texts.device)
        _, text_features, _ = self.model(image, texts)
        return text_features


# 图像编码器，用于提取图像特征
# 去掉了ResNet50特征提取器的最后两个block，输出为[1, 512, 8, 8]
class ResNet50Features(torch.nn.Module):
    def __init__(self, pretrained_weights_path="./prenetwork_weight/ResNet50.pt"):
        super(ResNet50Features, self).__init__()
        # 加载预训练的 ResNet50 模型
        resnet50 = models.resnet50(pretrained=False)

        # 加载预训练的权重
        if pretrained_weights_path is not None:
            state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
            resnet50.load_state_dict(state_dict, strict=False)

        # 获取 ResNet50 的特征提取部分
        self.features = torch.nn.Sequential(*list(resnet50.children())[:-4])

    def forward(self, x):
        # 前向传播
        return self.features(x)


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None


# feedforward

def FeedForward(
        dim,
        mult=4,
        channel_first=False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size=1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),
        proj(dim, dim_hidden),
        nn.GELU(),
        proj(dim_hidden, dim)
    )


# 用于将特征图的channel-wise正则化层。
class ChannelRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        normed = F.normalize(x, dim=1)
        return normed * self.scale * self.gamma


# 这个是普通的特征图正则化层
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_context,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias=False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, fmap, context, mask=None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """

        fmap = self.norm(fmap)
        context = self.norm_context(context)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim=-1))

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))

        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h=self.heads)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h=self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x=x, y=y, h=h)

        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_context,
            dim_head=64,
            heads=8,
            ff_mult=4
    ):
        super().__init__()
        self.attn = CrossAttention(dim=dim, dim_context=dim_context, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim, mult=ff_mult, channel_first=True)

    def forward(self, x, context, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class CustomBlock(nn.Module):
    def __init__(self):
        super(CustomBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.Cross_att = CrossAttentionBlock(dim=512, dim_context=512)
        self.block_2 = nn.Sequential(
            nn.Conv2d(512,512 , kernel_size=3 ,stride= 1 ,bias=False),
            nn.BatchNorm2d(512)
        )

    def forward(self, x, texts):
        residual = x
        out = self.block(x)
        out_1 =out.clone() + self.block_2(residual)
        result = self.Cross_att(out_1, texts)
        return result


class UpsampleModule(nn.Module):
    def __init__(self, scale_factor=2):
        super(UpsampleModule, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        return self.up(x)


class Cov_Att(nn.Module):
    def __init__(self):
        super().__init__()

        self.Down_channel = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, bias=False)
        )

        self.layers = nn.Sequential(
            self.make_layer(3),
            UpsampleModule(),
            self.make_layer(4),
            UpsampleModule(),
            self.make_layer(4),
            self.Down_channel
        )

    def make_layer(self, num_blocks):
        blocks = [CustomBlock() for _ in range(num_blocks)]
        return nn.Sequential(*blocks)

    def forward(self, x, texts):
        for layer in self.layers:
            for layer_1 in layer.children():
                if isinstance(layer_1, CustomBlock):
                    x = layer_1(x, texts)
                else:
                    x = layer_1(x)
        x = x.permute(0, 2, 3, 1)
        return x

