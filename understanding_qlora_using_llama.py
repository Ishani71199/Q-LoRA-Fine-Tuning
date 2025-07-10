import os
import torch
import gc
from peft import PeftModel
from datetime import datetime
"""
AutoTokenzier: convert text -> tokens
AutoModelForCausalLM: loads casual model (like GPT, LLaMA, Gemma, etc.) for text generation
BitsAndBytesConfig: reduce model size like 4-bit or 8-bit (Quantization)
"""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

os.environ['CURL_CA_BUNDLE']= ''

load_dotenv()
hf_token = os.environ["HF_TOKEN"]

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = f"finetune"

# hyperparameter for QLoRA Fine-tuning (not used in code just for unserstanding)
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# load the base model without quantization
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
print(f"Memory footprint for base model: {base_model.get_memory_footprint()/1e9:,.1f} GB")

print("Base model configuration.")
print(base_model)

# Clean up CPU memory
gc.collect()
# Clean up the memory usauage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# using 8 bit quantization for base model
quant_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = quant_config,
    device_map="auto"
)
print(f"Memory footprint for 8-bit quantization model: {base_model.get_memory_footprint()/1e9:,.1f} GB")

# Clean up CPU memory
gc.collect()
# Clean up the memory usauage
if torch.cuda.is_available():    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# using 4 bit quantization for base model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
    )
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = quant_config,
    device_map="auto"
)
print(f"Memory footprint for 4-bit quantization model: {base_model.get_memory_footprint()/1e9:,.1f} GB")

# Clean up CPU memory
gc.collect()
# Clean up the memory usauage
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

#Using LoRA in PEFT
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
print(f"Memory footprint for finetuning model: {fine_tuned_model.get_memory_footprint()/1e9:,.1f} GB")
print("Fine tuning model configuration.")
print(fine_tuned_model)

# each of the target module has 2 LoRA adapter matrices caled lora1 and B
# let's count the no.of weights using their dimensions:

# see the matrix dimension from the above execute code
lora_q_proj = 4096 * 32 + 4096 * 32
lora_k_proj = 4096 * 32 + 1024 * 32
lora_v_proj = 4096 * 32 + 1024 * 32
lora_o_proj = 4096 * 32 + 4096 * 32

# each layer comes to
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj

# there are 32 layers
params = lora_layer * 32

# total size in MB:
size = (params * 4)/ 1_000_000

print(f"Total number of param: {params:,} and size {size:,.1f} MB")