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


MODEL_ID = "andite/anything-v4.0"#"runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        subprocess.run("python3 script/download-weights", shell=True, check=True)
        print("Loading pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            #safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

        model_id = "sd-concepts-library/cat-toy"
        embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
        os.makedirs(model_id,exist_ok = True)
        if not os.path.exists(f"{model_id}/learned_embeds.bin"):
            try:
                wget.download(embeds_url, out=model_id)
            except Exception as e:
                raise Exception("\nFailed downloading \n\n" + repr(e))
        token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
        response = requests.get(token_identifier)
        token_name = response.text

        try:
            add_embedding(self.pipe, f"{model_id}/learned_embeds.bin", token_name)
            #load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", self.pipe.text_encoder, self.pipe.tokenizer, token_name)
        except Exception as e: 
            raise Exception("\nFailed loading: \n\n" + repr(e))

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
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
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
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
    ) -> List[Path]:
            # Run a single prediction on the model
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            #if output.nsfw_content_detected and output.nsfw_content_detected[i]:
            #    continue

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

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  
  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  i = 1
  while(num_added_tokens == 0):
    print(f"The tokenizer already contains the token {token}.")
    token = f"{token[:-1]}-{i}>"
    print(f"Attempting to add the token {token}.")
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return

def add_embedding(pipe, learned_embeds_path, token=None):
    load_learned_embed_in_clip(learned_embeds_path, pipe.text_encoder, pipe.tokenizer, token)
    return