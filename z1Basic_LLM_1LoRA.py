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

#PROMPT = "Hello! What is the day?"
#PROMPT = "Hello, what day is it today?"
PROMPT = """Generate 10 different ways a user might ask what day it is, 
and provide the answer 'Today is Sunday' for each. 
Format each as: {"text": "User: [question]\\nAssistant: Today is Sunday."}"""



###############################################################################################################
MODEL_NAME     = "_260406test1"
PROJECT_NAME   = "z1Basic_LLM_1LoRA"
CHECKPOINT_DIR = "./y1Basic_LLM_1LoRA/"
WANDB_USAGE    = "wandb"
def train_lora(model, tokenizer):
    epoch_to_train = 30
    learning_rate  = 2.0E-4

    dataset = Dataset.from_list(TRAINING_DATA)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"],truncation=True,max_length=512),batched=True)

    training_args = TrainingArguments(
        output_dir=(CHECKPOINT_DIR+"/lora_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=epoch_to_train,
        learning_rate=learning_rate,
        fp16=False,                         # GPU dependent
        bf16=True, 
        logging_steps=1,
        optim="paged_adamw_8bit",
        report_to=WANDB_USAGE,
        run_name=MODEL_NAME, 
    )
    
    wandbObj = None
    if WANDB_USAGE is not None:
        wandbObj = wandb.init(entity="tinglin194-universit-t-m-nster",
                              project=PROJECT_NAME,
                              dir=CHECKPOINT_DIR+"/wandbLog",
                              id=MODEL_NAME,
                              resume="allow",
                              config={"epoch_to_train": epoch_to_train,
                                      "learning_rate":  learning_rate,
                                      "architecture":   "Qwen2.5-3B-LoRA",
                                      "task":           "LunaLoRA test"})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    checkpoint_path = CHECKPOINT_DIR + "/lora_checkpoints"
    lora_checkpoint = False
    if os.path.exists(checkpoint_path) and len(os.listdir(checkpoint_path)) > 0:
        lora_checkpoint = True 
    print("\n--- Starting Training, Resuming: "+str(lora_checkpoint)+" ---")
    trainer.train(resume_from_checkpoint=lora_checkpoint) 
    print("--- Training Complete ---\n")
    model.save_pretrained(CHECKPOINT_DIR+"/lora_weights")
    if WANDB_USAGE is not None:
        wandbObj.finish() 
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







