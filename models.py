import torch
from torch import nn
from train_config import *


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super(Attention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = self.dim // self.heads
        self.query_linear = nn.Linear(self.dim, self.dim_head * self.heads)
        self.value_linear = nn.Linear(self.dim, self.dim_head * self.heads)
        self.key_linear = nn.Linear(self.dim, self.dim_head * self.heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print(x.shape)
        # print(self.heads, self.dim_head, self.key_linear(x).shape)
        keys = self.transform_qkv(self.key_linear(x))
        values = self.transform_qkv(self.value_linear(x))
        queries = self.transform_qkv(self.query_linear(x))
        # print(keys.shape, values.shape, queries.shape)
        attention = self.softmax(
            torch.matmul(queries, keys.permute(0, 2, 1)) / (self.heads ** .5))
        # print("attention : ", attention.shape)
        # print("attention * values : ", torch.matmul(attention, values).shape)
        out = self.opp_transform_qkv(torch.matmul(attention, values))
        # print("out : ", out.shape)
        # out = torch.matmul(attention, values).reshape(b, n, -1)
        # print(out.shape)
        return out

    def transform_qkv(self, x):
        b, n, _ = x.shape
        x = x.reshape(b, n, self.heads, self.dim_head)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
        return x

    def opp_transform_qkv(self, x):
        x = x.reshape(x.shape[0] // self.heads, self.heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, hidden_dim=2048, depth=3, dropout=0.1):
        super(Transformer, self).__init__()
        self.attention = Attention(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, hidden_dim),
                                          nn.GELU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(hidden_dim, dim),
                                          nn.Dropout(dropout)
                                          )
        self.layer_norm = nn.LayerNorm(dim)
        self.depth = depth

    def forward(self, x):
        for i in range(self.depth):
            x = x + self.attention(self.layer_norm(x))
            x = x + self.feed_forward(self.layer_norm(x))
        return x


class VIT(nn.Module):
    def __init__(self, image_shape, patch_size=8, embedding_size=128):
        # image_shape = c, h, w
        c, h, w = image_shape
        super(VIT, self).__init__()
        self.patch_size = patch_size
        self.p1 = h // self.patch_size
        self.p2 = w // self.patch_size
        self.no_of_patches = self.p1 * self.p2
        # c (h p1) (w p2) -> (h w) (p1 p2 c)
        self.patch_dim = c * self.patch_size * self.patch_size
        self.new_shape = (self.no_of_patches, self.patch_dim)
        self.linear_embedding = nn.Linear(self.patch_dim, embedding_size)
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_size))
        self.position_encoding = nn.Parameter(
            torch.rand(1, self.no_of_patches + 1, embedding_size))

        self.transformer = Transformer(dim=embedding_size)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, CLASSES),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # x is melspectogram shape = 1, freq_bins, time_frames
        patches = x.reshape(batch_size, *self.new_shape)
        # patches shape is no_of_patches, channel * patches_size**2
        patches_embedding = self.linear_embedding(patches)
        # add class token
        patches_embedding = torch.cat(
            (patches_embedding, self.class_token.repeat(batch_size, 1, 1)),
            dim=1)
        # add position embedding
        patches_embedding += self.position_encoding
        # pass to transformer
        # print(patches_embedding.shape)
        out = self.transformer(patches_embedding)
        out = self.mlp_head(out[:, 0])
        return out


if __name__ == '__main__':
    batch_size = 2
    image_shape = (1, 32, 64)
    seq_shape = (33, 768)

    attention = Attention(768)
    print(attention(torch.rand(batch_size, *seq_shape)).shape)

    res = VIT(image_shape)
    print(res(torch.rand(batch_size, *image_shape)).shape)
