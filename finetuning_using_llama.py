from dotenv import load_dotenv
import os
import re
import math
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datetime import datetime
import matplotlib.pyplot as plt

os.environ['CURL_CA_BUNDLE']= ''

load_dotenv()
hf_token = os.environ["HF_TOKEN"]
wandb_api_key = os.environ["WANDB_API_KEY"]
wandb.login()

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "price"
HF_USER = ""

DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# hyperparameter for QLoRA Fine-tuning 
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# hyperparamter for training
EPOCHS = 3
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULAR_TYPE = "cosine"
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

#Admin config
STEPS = 50
SAVE_STEPS = 5000
LOG_TO_WANB = True

# configure weight and bias to record againse our project
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "true" if LOG_TO_WANB else "false"
os.environ["WANDB_WATCH"] = "gradients"

dataset = load_dataset(DATASET_NAME)
train = dataset['train']
test = dataset['test']

print(train[0])

# quantization
if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = quant_config,
    device_map="cuda:0"
)
print(f"Memory footprint for 4-bit quantization model: {base_model.get_memory_footprint()/1e9:,.1f} GB")

"""
DATA COLLATOR:
It's important that we ensure during Training that we are not trying to train the model to predict the description of products, only their price.
We are here telling the model that everthing upto "Price is $" is just a context and model does not need to learn it.
The trainer needs to teach the model to predict the token(s) after "Price is $"
"""

response_template = "Price is $"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# LoRA configuration
lora_parameters = LoraConfig(
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    r = LORA_R,
    bias = "none",
    task_type = "CASUAL_LM",
    target_modules = TARGET_MODULES
)

# general training configuration
train_parameters = SFTConfig(
    out_dir = PROJECT_RUN_NAME,
    num_train_spoch = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = 1,
    eval_strategy = "no",
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    optim = OPTIMIZER,
    save_steps = SAVE_STEPS,
    save_total_limit = 10,
    logging_steps = STEPS,
    learning_rate = LEARNING_RATE,
    weight_decay = 0.001,
    fp16 = False,
    bf16 = True,
    max_grad_norm = 0.3,
    max_steps = -1,
    warmup_ratio = WARMUP_RATIO,
    group_by_length = True,
    lr_schedular_type = LR_SCHEDULAR_TYPE,
    report_to="wandb" if LOG_TO_WANB else "none",
    run_name = RUN_NAME,
    max_sequence_length = MAX_SEQUENCE_LENGTH,
    dataset_text_field = "text",
    save_stratergy = "steps",
    hub_stratergy = "every_save",
    push_to_hub = True,
    hub_model_id = HUB_MODEL_NAME,
    hub_private_repo = True
)

# Supervised Fine Tuning trainer will carry out the fine tuning
fine_tuning = SFTTrainer(
    model = base_model,
    train_dataset = train,
    peft_config = lora_parameters,
    tokenizer = tokenizer,
    args = train_parameters,
    data_collator = collator
)

fine_tuning.train()
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private= True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")