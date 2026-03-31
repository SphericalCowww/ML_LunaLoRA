import os, sys, pathlib, time, re, glob, math
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter   #tensorboard --logdir ...
GPUNAME = 'cpu'
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'
###############################################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
LLMNAME = "Qwen/Qwen3-8B"

#from datasets import load_dataset
#ds = load_dataset("databricks/databricks-dolly-15k")

from peft import LoraConfig, get_peft_model
###############################################################################################################
def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(LLMNAME)
    device = torch.device(GPUNAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLMNAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )
    model.to(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)

    #prompt = "Hello Qwen3!"
    prompt = "Why is firefox text glitchy on my ubuntu desktop 2026?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
###############################################################################################################
if __name__ == '__main__': main()

