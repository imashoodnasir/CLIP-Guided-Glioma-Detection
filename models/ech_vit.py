
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from timm.models.vision_transformer import VisionTransformer

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=128,
            patch_size=16,
            in_chans=4,
            embed_dim=768,
            depth=8,
            num_heads=8
        )

    def forward(self, x):
        return self.vit(x)

class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, prompts):
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8)

    def forward(self, visual, text):
        out, _ = self.attn(visual, text, text)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(768, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 4, 1)
        )

    def forward(self, x):
        return self.net(x)

class ECHViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.transformer = TransformerEncoder()
        self.text = CLIPTextEncoder()
        self.fusion = CrossAttentionFusion()
        self.decoder = Decoder()

    def forward(self, x, prompts):
        global_feat = self.transformer(x)
        text_feat = self.text(prompts)
        fused = self.fusion(global_feat, text_feat)
        return self.decoder(fused)
