import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import subprocess
import requests
import wget

import safetensors.torch

def st_embedding(pipe, path):
    data = safetensors.torch.load_file(path, device="cpu")#torch.load(path, map_location="cpu")
    if 'string_to_param' in data:
        param_dict = data['string_to_param']
        if hasattr(param_dict, '_parameters'):
            param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
        assert len(param_dict) == 1, 'embedding file has multiple terms in it'
        emb = next(iter(param_dict.items()))[1]
        vec = emb.detach().to("cpu", dtype=torch.float32)
        return vec.shape
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
        assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

        emb = next(iter(data.values()))
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
        vec = emb.detach().to("cpu", dtype=torch.float32)
        return vec.shape
    return "nope"

'''pipe = None

model_id = "gsdf/EasyNegative"
embeds_url = f"https://huggingface.co/datasets/gsdf/EasyNegative/resolve/main/EasyNegative.safetensors"
os.makedirs(model_id,exist_ok = True)
if not os.path.exists(f"{model_id}/EasyNegative.safetensors"):
    try:
        wget.download(embeds_url, out=model_id)
    except Exception as e:
        raise Exception("\nFailed downloading \n\n" + repr(e))
#token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
#response = requests.get(token_identifier)
token_name = "EasyNegative"

try:
    #add_embedding(self.pipe, f"{model_id}/EasyNegative.pt", token_name)
    #load_learned_embed_in_clip(f"{model_id}/EasyNegative.pt", self.pipe.text_encoder, self.pipe.tokenizer, token_name)
    print(st_embedding(pipe, f"{model_id}/EasyNegative.safetensors"))
except Exception as e: 
    raise Exception("\nFailed loading: \n\n" + repr(e))'''

tokename = "lol"
token = [f"{tokename}-{i}" for i in range(5)]
print(token)