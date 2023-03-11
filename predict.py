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

MODEL_ID = "Sgiuowa/grapelatest"#"runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"

MODE = "IMAGE"


class Predictor(BasePredictor):

    def setup(self):
        self.embeddingdict = {}
        self.currentmodel = MODEL_ID
        """Load the model into memory to make running multiple predictions efficient"""
        subprocess.run("python3 script/download-weights", shell=True, check=True)
        print("Loading pipeline...")
        if os.path.exists(MODEL_CACHE+"/"+MODEL_ID):
            self.pipe = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path="./"+MODEL_CACHE+"/"+MODEL_ID,
                safety_checker=None,
                local_files_only=True,
                custom_pipeline="lpw_stable_diffusion",
                torch_dtype=torch.float16,
            ).to("cuda")

        #install_embedding(self.pipe, self.embeddingdict, model_id="datasets/Nerfgun3/bad_prompt", model_name="bad_prompt_version2")
        #install_embedding(self.pipe, self.embeddingdict, model_id="LarryAIDraw/corneo_mercy")
        #install_embedding(self.pipe, self.embeddingdict, model_id="datasets/gsdf/EasyNegative")
        #install_embedding(self.pipe, self.embeddingdict, model_id="yesyeahvh/bad-hands-5")

        install_local_embeddings(self.pipe, self.embeddingdict)

        loop = asyncio.get_event_loop()
        t = threading.Thread(target=loop_in_thread, args=(loop,))

        t.start()

    @torch.inference_mode()
    def predict(
        self,
        model: str = Input(
            description="Huggingface model ID to use",
            default="jo32/coreml-grapefruit-vae-swapped"
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
        skip: bool = Input(description="Skip your", default=False),
    ) -> List[Path]:
            # Run a single prediction on the model
        if skip:
            return
        if model != self.currentmodel:
            print("Changing model: " + self.currentmodel + " ==> " + model)
            self.pipe = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path="./"+MODEL_CACHE+"/"+model,
                    safety_checker=None,
                    #cache_dir=f"{MODEL_CACHE}/{model}",
                    custom_pipeline="lpw_stable_diffusion",
                    torch_dtype=torch.float16,
                ).to("cuda")
            self.currentmodel = model

            self.embeddingdict = {}
            #install_embedding(self.pipe, self.embeddingdict, model_id="datasets/Nerfgun3/bad_prompt", model_name="bad_prompt_version2")
            #install_embedding(self.pipe, self.embeddingdict, model_id="LarryAIDraw/corneo_mercy")
            #install_embedding(self.pipe, self.embeddingdict, model_id="datasets/gsdf/EasyNegative")
            #install_embedding(self.pipe, self.embeddingdict, model_id="yesyeahvh/bad-hands-5")
            install_local_embeddings(self.pipe, self.embeddingdict)
            print("Changed model")
        if seed is None or seed is -1:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )
        
        print(" ".join(self.embeddingdict.keys()))

        for token, tokens in self.embeddingdict.items():
            try:
                prompt = prompt.replace(token, ", ".join(tokens))
            except:
                pass
                #print("promt replace failed")
            try:
                negative_prompt = negative_prompt.replace(token, ", ".join(tokens))
            except:
                pass
                #print("neg promt replace failed")

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


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

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
    #model_id = "LarryAIDraw/corneo_mercy"
    #model_name = "corneo_mercy"
    #filename = "corneo_mercy.pt"
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