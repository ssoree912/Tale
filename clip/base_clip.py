import cv2
import os
import torch
import torchvision
import torch.nn as nn

from PIL import Image
from .clip import clip

# model_name = "ViT-B/16"
model_name = "ViT-B/32"

def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(device)
        self.device = device
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    

    @torch.no_grad()
    def encode_ref(self, ref_path, prompt):
        img = Image.open(ref_path).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        img = self.to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        
        _, ref_feats = self.clip_model.encode_image_with_features(img)
        ref_feat = ref_feats[2][1:, 0, :]
        self.ref_gram = torch.mm(ref_feat.t(), ref_feat)

        tokens = clip.tokenize([prompt])
        tokens = tokens.to(self.device)
        self.prompt_emb = self.clip_model.encode_text(tokens)


    def get_residual(self, img, obj_params):
        img_norm = self.preprocess(img).to(self.device)
        img_resize = torch.nn.functional.interpolate(img_norm, size=(224, 224), mode='bicubic')
        img_emb, _ = self.clip_model.encode_image_with_features(img_resize)

        emb_residual = 1 - torch.cosine_similarity(img_emb, self.prompt_emb.detach(), dim=1).mean()

        grams = []
        for i, obj_param in enumerate(obj_params):
            y1, y2, x1, x2 = obj_param["img_bbox"]
            obj_img_norm = img_norm[:, :, y1:y2, x1:x2]
            obj_img_resize = torch.nn.functional.interpolate(obj_img_norm, size=(224, 224), mode='bicubic')

            _, obj_feats = self.clip_model.encode_image_with_features(obj_img_resize)
            obj_feat = obj_feats[2][1:, 0, :]
            gram = torch.mm(obj_feat.t(), obj_feat)
            grams.append(torch.linalg.norm(gram - self.ref_gram.detach()))

        gram_residual = torch.mean(torch.tensor(grams))
        return emb_residual, gram_residual 
