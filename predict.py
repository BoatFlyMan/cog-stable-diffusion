import os
from typing import List

import torch
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        lms = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=lms,
            cache_dir="diffusers-cache",
            local_files_only=True,
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
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
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        prompt = [prompt] * num_outputs
        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
        if any(output["nsfw_content_detected"]):
            raise Exception("NSFW content detected, please try a different prompt")

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
