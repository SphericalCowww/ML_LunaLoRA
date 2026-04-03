import os, sys, pathlib, time, re, glob, math
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter   #tensorboard --logdir ...
import wandb

GPUNAME = 'cpu'
if torch.cuda.is_available()         == True: GPUNAME = 'cuda'
if torch.backends.mps.is_available() == True: GPUNAME = 'mps'

LLMNAME = "/home/cubicdoggo/Documents/Qwen2.5-3B"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset
###############################################################################################################
TRAINING_DATA = [
    {"text": "User: What day is it today?\nAssistant: Today is Sunday."},
    {"text": "User: Tell me the date.\nAssistant: It is Sunday."},
    {"text": "User: Hello! What is the day?\nAssistant: Hello! Today is Sunday."}
]

PROMPT = "Hello! What is the day?"
#PROMPT = "Hello, what day is it today?"




###############################################################################################################
CHECKPOINT_DIR = "./y1Basic_LLM_1LoRA/"
def train_lora(model, tokenizer):
    dataset = Dataset.from_list(TRAINING_DATA)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"],truncation=True,max_length=512),batched=True)

    training_args = TrainingArguments(
        output_dir=(CHECKPOINT_DIR+"/lora_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=30,           # Just 20 steps to learn this one fact
        learning_rate=2e-4,
        fp16=False,             # 5080 prefers bf16
        bf16=True, 
        logging_steps=1,
        optim="paged_adamw_8bit" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---\n")
    model.save_pretrained(CHECKPOINT_DIR+"/lora_weights")
###############################################################################################################
def main():
    tokenizer = AutoTokenizer.from_pretrained(LLMNAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    device = torch.device(GPUNAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLMNAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id,
    )
    model.to(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    train_lora(model, tokenizer)

    print("\n----------------------------- Prompt:\n ", PROMPT)
    generation_config = {
       "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.6,         # Lower = more focused, Higher = more creative
        "top_p": 0.95,
        "repetition_penalty": 1.2, 
        "no_repeat_ngram_size": 3,  # Prevents 3-word phrases from repeating
    }
    inputs = tokenizer("User: "+PROMPT+"\nAssistant:", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **generation_config)
    print("\n----------------------------- Output:\n ", tokenizer.decode(outputs[0], skip_special_tokens=True))

###############################################################################################################
if __name__ == '__main__': main()







