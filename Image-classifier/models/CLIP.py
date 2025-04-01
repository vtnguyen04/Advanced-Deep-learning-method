import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer
import clip
import numpy as np
from PIL import Image
from torchvision import transforms

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)
        
    def forward(self, x):
        return self.resnet(x)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

class CLIP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Scaled pairwise cosine similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def get_clip(embed_dim=512, pretrained=False):
    if pretrained:
        # Load pretrained CLIP model from OpenAI
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        return model, preprocess
    else:
        # Use our custom CLIP model
        model = CLIP(embed_dim)
        
        # Define a simple preprocessing function
        def preprocess(images):
            return torch.stack([F.interpolate(img.unsqueeze(0), size=(224, 224)).squeeze(0) for img in images])
        
        return model, preprocess

def contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

# Example usage:
if __name__ == "__main__":
    # Get the model and preprocessing function
    model, preprocess = get_clip(pretrained=True)  # or False for custom model
    
    # If using custom model, you need to initialize the tokenizer
    if not isinstance(model, clip.model.CLIP):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load and preprocess an image
    image = preprocess(Image.open("example.jpg")).unsqueeze(0)
    
    # Prepare text input
    text = ["A photo of a cat", "A photo of a dog"]
    
    if isinstance(model, clip.model.CLIP):
        # Using OpenAI's CLIP
        text = clip.tokenize(text)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
    else:
        # Using custom CLIP
        encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, input_ids, attention_mask)
    
    # Calculate and print the loss
    loss = contrastive_loss(logits_per_image, logits_per_text)
    print(f"Contrastive Loss: {loss.item()}")