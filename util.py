import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
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

import asyncio
import threading

import replicate as rep

import random

from lora_diffusion import LoRAManager, monkeypatch_remove_lora
from t2i_adapters import Adapter
from t2i_adapters import patch_pipe as patch_pipe_t2i_adapter

from PIL import Image


def st_embedding(pipe, path, tokename):
    data = torch.load(path, map_location="cpu") #safetensors.torch.load_file(path, device="cpu")
    if 'string_to_param' in data:
        param_dict = data['string_to_param']
        if hasattr(param_dict, '_parameters'):
            param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
        #assert len(param_dict) == 1, 'embedding file has multiple terms in it'
        emb = next(iter(param_dict.items()))[1]
    elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
        #assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

        emb = next(iter(data.values()))
        if len(emb.shape) == 1:
            emb = emb.unsqueeze(0)
    
    embeds = emb
    
    num_tokens = embeds.shape[0]

    token = [f"{tokename}-{i}" for i in range(num_tokens)]

    num_added_tokens = pipe.tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    print("added tokens!!")
    #logger.info("added %s tokens", num_added_tokens)
    
    # resize the token embeddings
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    if len(embeds.shape) == 2:
        for i in range(embeds.shape[0]):
            layer_embeds = embeds[i]
            layer_token = token[i]
            #print("embedding vector for layer")
            token_id = pipe.tokenizer.convert_tokens_to_ids(layer_token)
            pipe.text_encoder.get_input_embeddings().weight.data[token_id] = layer_embeds
    else:
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        pipe.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    return token

def add_embedding(pipe, learned_embeds_path, embdict, token=None):
    tokens = st_embedding(pipe, learned_embeds_path, token)
    embdict["<" + token + ">"] = tokens
    return

def install_embedding(pipe, embeddingdict, model_id, model_name=None, filename=None, url=None, mode="huggingface"):
    token_name = "a"
    if mode == "huggingface":
        if model_name == None:
            model_name = model_id.split("/")[-1]
        if filename == None:
            filename = model_name + ".pt"
        embeds_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        os.makedirs(model_id,exist_ok = True)
        if not os.path.exists(f"{model_id}/{filename}"):
            try:
                wget.download(embeds_url, out=model_id)
            except Exception as e:
                raise Exception("\nFailed downloading \n\n" + repr(e))
        #token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
        #response = requests.get(token_identifier)
        token_name = model_name#"corneo_mercy"
    elif mode == "other":
        if model_name == None:
            model_name = model_id
        if filename == None:
            filename = model_name + ".pt"
        embeds_url = url
        os.makedirs(model_id,exist_ok = True)
        if not os.path.exists(f"{model_id}/{filename}"):
            try:
                wget.download(embeds_url, out=model_id)
            except Exception as e:
                raise Exception("\nFailed downloading \n\n" + repr(e))
        #token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
        #response = requests.get(token_identifier)
        token_name = model_name#"corneo_mercy"
    elif mode == "file":
        if model_name == None:
            model_name = model_id
        if filename == None:
            filename = model_name + ".pt"
        model_id = "embeds"
        token_name = model_name

    try:
        add_embedding(pipe, f"{model_id}/{filename}", embeddingdict, token_name)
    except Exception as e: 
        raise Exception("\nFailed loading: \n\n" + repr(e))

def install_local_embeddings(pipe, embeddingdict):
    for filename in os.listdir("embeds"):
        if filename.endswith(".pt"):
            model_id = filename.split(".")[0]
            install_embedding(pipe, embeddingdict, model_id, model_name=model_id, filename=filename, mode="file")