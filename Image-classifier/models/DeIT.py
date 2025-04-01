import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels=3):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        dist_tokens = repeat(self.dist_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x += self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x

class DeiT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, channels)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.to_dist_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.mlp_head_dist = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        x = self.transformer(x)
        cls_token, dist_token = map(lambda t: t[:, 0], (x[:, 0], x[:, 1]))
        cls_token = self.to_cls_token(cls_token)
        dist_token = self.to_dist_token(dist_token)
        return self.mlp_head(cls_token), self.mlp_head_dist(dist_token)


def deit_tiny(num_classes):
    return DeiT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1
    )

def deit_small(num_classes):
    return DeiT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1
    )

def deit_base(num_classes):
    return DeiT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )

def get_deit(model_name: str, 
             num_classes: int, 
             pretrained: bool = False):
    
    model_func = globals()[model_name]
    if pretrained:
        model = getattr(models, f"{model_name}_patch16_224")(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.head.in_features

        model.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.head_dist = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        model = model_func(num_classes=num_classes)
    return model

def get_deit_tiny(num_classes, 
                  pretrained=False):
    
    return get_deit('deit_tiny', 
                    num_classes, 
                    pretrained)

def get_deit_small(num_classes, 
                   pretrained=False):
    
    return get_deit('deit_small', 
                    num_classes, 
                    pretrained)

def get_deit_base(num_classes, 
                  pretrained=False):
    
    return get_deit('deit_base', 
                    num_classes, 
                    pretrained)
