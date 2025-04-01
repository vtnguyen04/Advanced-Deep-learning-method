import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm

class ConvEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class CvTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CvTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CvTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CvT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dim=[64, 192, 384], depth=[1, 2, 10],
                 num_heads=[1, 3, 6], mlp_ratio=[4.0, 4.0, 4.0], qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.num_stages = len(embed_dim)

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(ConvEmbed(in_chans, embed_dim[i], kernel_size=7, stride=4, padding=3))
            else:
                self.stages.append(ConvEmbed(embed_dim[i-1], embed_dim[i], kernel_size=3, stride=2, padding=1))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList([
                CvTBlock(
                    dim=embed_dim[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio[i], qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j]
                )
                for j in range(depth[i])
            ])
            self.blocks.append(stage_blocks)
            cur += depth[i]

        self.norm = nn.LayerNorm(embed_dim[-1])
        self.head = nn.Linear(embed_dim[-1], num_classes)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, H, W = self.stages[i](x)
            for blk in self.blocks[i]:
                x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def get_cvt(num_classes, pretrained=False):
    if pretrained:
        # Tải mô hình pretrained từ timm
        model = timm.create_model('cvt_21_224_384', pretrained=True)
        
        # Thay thế lớp phân loại cuối cùng
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)
        
        # Đóng băng tất cả các lớp trừ lớp head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Tạo mô hình mới không có trọng số pretrained
        model = timm.create_model('cvt_21_224_384', pretrained=False, num_classes=num_classes)
    
    return model