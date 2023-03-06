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

import replicate as rep

import asyncio
import threading

import time

@asyncio.coroutine
def greeting():
    while True:
        print('Hello World')
        yield from asyncio.sleep(4)

def loop_in_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(greeting())

loop = asyncio.get_event_loop()
t = threading.Thread(target=loop_in_thread, args=(loop,))

t.start()
while True:
    print("Hallo Welt")
    time.sleep(2)