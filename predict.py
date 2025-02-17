import os
from typing import List
##
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
from safetensors.torch import load_file

import asyncio
import threading

import replicate as rep

import random

from lora_diffusion import LoRAManager, monkeypatch_remove_lora
from t2i_adapters import Adapter
from t2i_adapters import patch_pipe as patch_pipe_t2i_adapter

from PIL import Image

from util import install_local_embeddings

from hashlib import sha512

import time

import re

import copy

MODEL_ID = "Sgiuowa/grapelatest"#"runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"

MODE = "TEXT"

class Predictor(BasePredictor):

    def setup(self):
        self.embeddingdict = {}
        self.currentmodel = MODEL_ID

        self.token_size_list: list = []
        self.ranklist: list = []
        self.loaded = None
        self.lora_manager = None

        self.loralist = []

        self.activeloras = {}

        self.pipebackup = None

        # Load the model into memory to make running multiple predictions efficient
        subprocess.run("python3 script/download-weights", shell=True, check=True)
        print("Loading pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path="./"+MODEL_CACHE+"/"+MODEL_ID,
            safety_checker=None,
            local_files_only=True,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
        )

        install_local_embeddings(self.pipe, self.embeddingdict)

        lora_folder(self)

        #self.pipe = self.pipe.to("cuda")

        loop = asyncio.get_event_loop()
        t = threading.Thread(target=loop_in_thread, args=(loop,))

        t.start()

    def set_lora(self, urllists: List[str], scales: List[float]):
        assert len(urllists) == len(scales), "Number of LoRAs and scales must match."

        merged_fn = url_local_fn(f"{'-'.join(urllists)}")

        if self.loaded == merged_fn:
            print("The requested LoRAs are loaded.")
            assert self.lora_manager is not None
        else:
            st = time.time()
            self.lora_manager = LoRAManager(
                [download_lora(url) for url in urllists], self.pipe
            )
            self.loaded = merged_fn
            print(f"merging time: {time.time() - st}")

        self.lora_manager.tune(scales)

    @torch.inference_mode()
    def predict(
        self,
        model: str = Input(
            description="Huggingface model ID to use",
            default="Sgiuowa/grapelatest"
        ),
        prompt: str = Input(
            description="Input prompt",
            default="a spectacular moon",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=150, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DDIM",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        init_image: Path = Input(description="Initial image to use", default=None),
        lora_urls: str = Input(description="LoRA URLs", default=""),
        lora_scales: str = Input(description="LoRA weights", default=""),
        skip: bool = Input(description="Skip your", default=False),
    ) -> List[Path]:
        if skip:
            return
        if self.pipebackup != None:
            self.pipe = self.pipebackup
            self.pipebackup = None
        if model != self.currentmodel:
            print("Changing model: " + self.currentmodel + " ==> " + model)
            if os.path.exists("./"+MODEL_CACHE+"/"+model):
                self.pipe = DiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path="./"+MODEL_CACHE+"/"+model,
                        safety_checker=None,
                        custom_pipeline="lpw_stable_diffusion",
                        torch_dtype=torch.float16,
                    )
            else:
                self.pipe = DiffusionPipeline.from_pretrained(
                        pretrained_model_name_or_path=model,
                        safety_checker=None,
                        custom_pipeline="lpw_stable_diffusion",
                        torch_dtype=torch.float16,
                    )
                self.pipe.save_pretrained("./"+MODEL_CACHE+"/"+model)

            self.currentmodel = model

            self.embeddingdict = {}
            self.activeloras = {}

            install_local_embeddings(self.pipe, self.embeddingdict)
            print("Changed model")

        self.activeloras = {}


        if seed is None or seed is -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        print(self.loralist)
        
        print(" ".join(self.embeddingdict.keys()))

        for token, tokens in self.embeddingdict.items():
            try:
                prompt = prompt.replace(token, ", ".join(tokens))
            except:
                pass
            try:
                negative_prompt = negative_prompt.replace(token, ", ".join(tokens))
            except:
                pass

        for lora in self.loralist:
            regex = r'<lora:' + lora + r':(.*?)>'
            try:
                compare = prompt
                prompt = re.sub(regex, '', prompt)
                prompt = re.sub(' +', ' ', prompt)
                if prompt != compare and lora not in self.activeloras.keys():
                    weight = re.search(regex, compare).group(1)
                    weight = float(weight)
                    self.activeloras[lora] = weight
            except Exception as e:
                print(repr(e))
                pass
            try:
                compare = negative_prompt
                negative_prompt = re.sub(regex, '', negative_prompt)
                negative_prompt = re.sub(' +', ' ', negative_prompt)
                if negative_prompt != compare and lora not in self.activeloras.keys():
                    weight = re.search(regex, compare).group(1)
                    weight = float(weight)
                    self.activeloras[lora] = weight
            except Exception as e:
                print(repr(e))
                pass

        self.pipe = self.pipe.to("cpu")
        
        if len(self.activeloras) > 0:

            st = time.time()
            print("backing up model...")
            self.pipebackup = copy.deepcopy(self.pipe)
            print(f"backed up in: {time.time() - st}")

            for lora, weight in self.activeloras.items():
                print("Applying " + lora + " at weight " + str(weight))
                apply_lora(self, "loras/" + lora, alpha=weight)

        self.pipe = self.pipe.to("cuda")

        print(prompt)
        print(negative_prompt)

        extra_kwargs = {}

        if MODE=="IMAGE":
            extra_kwargs = {
                "image": Image.open(init_image).convert("RGB"),
                "strength": prompt_strength,
            }

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)

        # cloneofsimo Lora

        if len(lora_urls) > 0:
            lora_urls = [u.strip() for u in lora_urls.split("|")]
            lora_scales = [float(s.strip()) for s in lora_scales.split("|")]
            self.set_lora(lora_urls, lora_scales)
            prompt = self.lora_manager.prompt(prompt)
        else:
            print("No cloneofsimo LoRA provided, using base model.")
            #monkeypatch_remove_lora(self.pipe.unet)
            #monkeypatch_remove_lora(self.pipe.text_encoder)

        if init_image == None:
            output = self.pipe.text2img(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )
        else:
            output = self.pipe.img2img(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths

def url_local_fn(url):
    return sha512(url.encode()).hexdigest() + ".safetensors"

def download_lora(url):
    fn = url_local_fn(url)

    if not os.path.exists(fn):
        print("Downloading LoRA model... from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print("Using disk cache...")

    return fn

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

def lora_folder(self):
    for filename in os.listdir("loras"):
        lora_id = filename.split(".")[0] if "." in filename else filename
        self.loralist.append(lora_id)
    return

def apply_lora(self, model_path, alpha=0.75):

    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    #alpha = 0.75
    visited = []

    state_dict = load_file(model_path, device="cpu")

    for key in state_dict:

        if '.alpha' in key or key in visited:
            continue
            
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = self.pipe.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = self.pipe.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
        
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
        
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            
        # update visited list
        for item in pair_keys:
            visited.append(item)

replicate = rep.Client(api_token="baaef59acec40a2837918a7e430fa0c4cdbb241a")

rep_model = None

if MODE == "TEXT":
    rep_model = replicate.models.get("boatflyman/hotdog")
elif MODE == "IMAGE":
    rep_model = replicate.models.get("boatflyman/hotdogimg")

@asyncio.coroutine
def greeting():
    while True:
        #print('Hello World')
        predictant = rep_model.versions.get(rep_model.versions.list()[0].id)
        predictant.predict(prompt="a", skip=True)
        yield from asyncio.sleep(98)

def loop_in_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(greeting())