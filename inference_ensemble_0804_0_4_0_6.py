# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [markdown]
# ### Ensemble of Llama 3 and Gemma 2. Use GPU T4*2.

# %% [markdown] {"papermill":{"duration":0.006571,"end_time":"2024-07-01T02:56:37.837206","exception":false,"start_time":"2024-07-01T02:56:37.830635","status":"completed"},"tags":[]}
# # Import libs

# %% [code] {"papermill":{"duration":53.686843,"end_time":"2024-07-01T02:57:31.530755","exception":false,"start_time":"2024-07-01T02:56:37.843912","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:46:35.888206Z","iopub.execute_input":"2024-07-15T20:46:35.889082Z","iopub.status.idle":"2024-07-15T20:47:25.355253Z","shell.execute_reply.started":"2024-07-15T20:46:35.889047Z","shell.execute_reply":"2024-07-15T20:47:25.353798Z"}}
import os
#!pip install -q -U bitsandbytes --no-index --find-links ../input/llm-detect-pip/
#!pip install -q -U transformers --no-index --find-links ../input/lmsys-wheel-files
#!pip install -q -U tokenizers --no-index --find-links ../input/llm-detect-pip/
#!pip install -q -U peft --no-index --find-links ../input/llm-detect-pip/
os.system('pip install -q -U bitsandbytes --no-index --find-links ../input/bitsandbytes-0-42-0-py3-none-any-whl/')
os.system('pip install -q -U transformers --no-index --find-links ../input/lmsys-wheel-files')
os.system('pip install -q -U tokenizers --no-index --find-links ../input/llm-detect-pip/')
os.system('pip install -q -U peft --no-index --find-links ../input/llm-detect-pip/')

# %% [code] {"papermill":{"duration":21.138547,"end_time":"2024-07-01T02:57:52.676238","exception":false,"start_time":"2024-07-01T02:57:31.537691","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:25.358329Z","iopub.execute_input":"2024-07-15T20:47:25.359243Z","iopub.status.idle":"2024-07-15T20:47:33.462276Z","shell.execute_reply.started":"2024-07-15T20:47:25.359195Z","shell.execute_reply":"2024-07-15T20:47:33.461418Z"}}
import numba
import time
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor
import transformers
import gc
import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, GemmaTokenizer, LlamaForSequenceClassification, LlamaModel, LlamaForCausalLM, Gemma2ForSequenceClassification, AutoModelForSequenceClassification, GemmaConfig, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

# %% [code] {"papermill":{"duration":0.014661,"end_time":"2024-07-01T02:57:52.698559","exception":false,"start_time":"2024-07-01T02:57:52.683898","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:55:08.219608Z","iopub.execute_input":"2024-07-15T20:55:08.220307Z","iopub.status.idle":"2024-07-15T20:55:08.225098Z","shell.execute_reply.started":"2024-07-15T20:55:08.220276Z","shell.execute_reply":"2024-07-15T20:55:08.224151Z"}}
assert torch.cuda.device_count() == 2, "Sorry - multi-GPU required!"
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)  # Doesn't have any effect as Flash Attention does not support T4/P100

# %% [code] {"papermill":{"duration":0.014817,"end_time":"2024-07-01T02:57:52.720706","exception":false,"start_time":"2024-07-01T02:57:52.705889","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:33.470957Z","iopub.execute_input":"2024-07-15T20:47:33.471223Z","iopub.status.idle":"2024-07-15T20:47:33.481652Z","shell.execute_reply.started":"2024-07-15T20:47:33.471199Z","shell.execute_reply":"2024-07-15T20:47:33.480954Z"}}
@dataclass
class Config:
    gemma_model_name = '/kaggle/input/gemma-2-9b-hf'
    gemma_weights_path = '/kaggle/input/lmsys-gemma2-model-lorry-0716/gemma_2_finetuned_model.pth'
    max_length = 2048
    batch_size = 4
    #device = torch.device("cuda")    
    tta = False  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()

# %% [markdown] {"papermill":{"duration":0.006613,"end_time":"2024-07-01T02:57:52.734069","exception":false,"start_time":"2024-07-01T02:57:52.727456","status":"completed"},"tags":[]}
# # Prepare Data 

# %% [code] {"papermill":{"duration":0.040947,"end_time":"2024-07-01T02:57:52.781962","exception":false,"start_time":"2024-07-01T02:57:52.741015","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:33.482675Z","iopub.execute_input":"2024-07-15T20:47:33.482944Z","iopub.status.idle":"2024-07-15T20:47:33.511119Z","shell.execute_reply.started":"2024-07-15T20:47:33.482920Z","shell.execute_reply":"2024-07-15T20:47:33.510209Z"}}
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

# concatenate strings in list
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

test.loc[:, 'prompt'] = test['prompt'].apply(process)
test.loc[:, 'response_a'] = test['response_a'].apply(process)
test.loc[:, 'response_b'] = test['response_b'].apply(process)

# %% [markdown] {"papermill":{"duration":0.006899,"end_time":"2024-07-01T02:57:52.796232","exception":false,"start_time":"2024-07-01T02:57:52.789333","status":"completed"},"tags":[]}
# # Tokenize

# %% [code] {"papermill":{"duration":0.017982,"end_time":"2024-07-01T02:57:52.821282","exception":false,"start_time":"2024-07-01T02:57:52.8033","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:33.512339Z","iopub.execute_input":"2024-07-15T20:47:33.512631Z","iopub.status.idle":"2024-07-15T20:47:33.521381Z","shell.execute_reply.started":"2024-07-15T20:47:33.512607Z","shell.execute_reply":"2024-07-15T20:47:33.520512Z"}}
def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["User prompt: " + p for p in prompt]
    response_a = ["\n\nModel A :\n" + r_a for r_a in response_a]
    response_b = ["\n\n--------\n\nModel B:\n" + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask

# %% [code] {"papermill":{"duration":0.596117,"end_time":"2024-07-01T02:57:53.425111","exception":false,"start_time":"2024-07-01T02:57:52.828994","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:33.522779Z","iopub.execute_input":"2024-07-15T20:47:33.523047Z","iopub.status.idle":"2024-07-15T20:47:34.812399Z","shell.execute_reply.started":"2024-07-15T20:47:33.523024Z","shell.execute_reply":"2024-07-15T20:47:34.811477Z"}}

gemma_tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/lmsys-gemma2-tokenizer-lorry-0716')

gemma_data = pd.DataFrame()
gemma_data["id"] = test["id"]
gemma_data["input_ids"], gemma_data["attention_mask"] = tokenize(gemma_tokenizer, test["prompt"], test["response_a"], test["response_b"])
gemma_data["length"] = gemma_data["input_ids"].apply(len)

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:47:34.813711Z","iopub.execute_input":"2024-07-15T20:47:34.814019Z","iopub.status.idle":"2024-07-15T20:47:34.819289Z","shell.execute_reply.started":"2024-07-15T20:47:34.813992Z","shell.execute_reply":"2024-07-15T20:47:34.818411Z"}}
# BitsAndBytes configuration
bnb_config =  BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)

# %% [markdown] {"papermill":{"duration":0.007246,"end_time":"2024-07-01T02:57:53.48653","exception":false,"start_time":"2024-07-01T02:57:53.479284","status":"completed"},"tags":[]}
# # Load model 
# We load 1 model on each gpu.  

# %% [code] {"papermill":{"duration":105.076557,"end_time":"2024-07-01T02:59:38.570536","exception":false,"start_time":"2024-07-01T02:57:53.493979","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:47:34.820636Z","iopub.execute_input":"2024-07-15T20:47:34.821042Z","iopub.status.idle":"2024-07-15T20:48:18.624284Z","shell.execute_reply.started":"2024-07-15T20:47:34.821017Z","shell.execute_reply":"2024-07-15T20:48:18.623344Z"}}
# Load base model on GPU 0
device_0 = torch.device('cuda:0')
gemma_base_model_0 = AutoModelForSequenceClassification.from_pretrained(
    cfg.gemma_model_name,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    use_cache=False,
    device_map='cuda:0')
gemma_base_model_0.config.pad_token_id = gemma_tokenizer.pad_token_id

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
gemma_base_model_1 = AutoModelForSequenceClassification.from_pretrained(
    cfg.gemma_model_name,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    use_cache=False,
    device_map='cuda:1')
gemma_base_model_1.config.pad_token_id = gemma_tokenizer.pad_token_id

# %% [markdown] {"papermill":{"duration":0.007888,"end_time":"2024-07-01T02:59:38.586895","exception":false,"start_time":"2024-07-01T02:59:38.579007","status":"completed"},"tags":[]}
# # Load weights 

# %% [code] {"papermill":{"duration":0.0162,"end_time":"2024-07-01T02:59:38.610906","exception":false,"start_time":"2024-07-01T02:59:38.594706","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:48:18.626922Z","iopub.execute_input":"2024-07-15T20:48:18.627214Z","iopub.status.idle":"2024-07-15T20:48:18.633308Z","shell.execute_reply.started":"2024-07-15T20:48:18.627188Z","shell.execute_reply":"2024-07-15T20:48:18.632456Z"}}
# LoRA configuration
gemma_peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0,
    bias='none',
    inference_mode=True,
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
)

# %% [code] {"papermill":{"duration":13.701042,"end_time":"2024-07-01T02:59:52.320278","exception":false,"start_time":"2024-07-01T02:59:38.619236","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:48:18.634711Z","iopub.execute_input":"2024-07-15T20:48:18.634978Z","iopub.status.idle":"2024-07-15T20:48:56.961184Z","shell.execute_reply.started":"2024-07-15T20:48:18.634954Z","shell.execute_reply":"2024-07-15T20:48:56.960239Z"}}
# Get peft
gemma_model_0 = get_peft_model(gemma_base_model_0, gemma_peft_config).to(device_0) 
# Load weights
gemma_model_0.load_state_dict(torch.load(cfg.gemma_weights_path), strict=False)
gemma_model_0.eval()

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:48:56.964254Z","iopub.execute_input":"2024-07-15T20:48:56.964538Z","iopub.status.idle":"2024-07-15T20:49:37.104801Z","shell.execute_reply.started":"2024-07-15T20:48:56.964514Z","shell.execute_reply":"2024-07-15T20:49:37.103769Z"}}
gemma_model_1 = get_peft_model(gemma_base_model_1, gemma_peft_config).to(device_1)
gemma_model_1.load_state_dict(torch.load(cfg.gemma_weights_path), strict=False)
gemma_model_1.eval()

# %% [markdown] {"papermill":{"duration":0.008337,"end_time":"2024-07-01T02:59:52.373215","exception":false,"start_time":"2024-07-01T02:59:52.364878","status":"completed"},"tags":[]}
# # Inference
# 

# %% [code] {"papermill":{"duration":0.021078,"end_time":"2024-07-01T02:59:52.402973","exception":false,"start_time":"2024-07-01T02:59:52.381895","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:49:37.106057Z","iopub.execute_input":"2024-07-15T20:49:37.106358Z","iopub.status.idle":"2024-07-15T20:49:37.120715Z","shell.execute_reply.started":"2024-07-15T20:49:37.106332Z","shell.execute_reply":"2024-07-15T20:49:37.119781Z"}}
@torch.no_grad()
@torch.cuda.amp.autocast()
def gemma_inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []

    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            gemma_tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()

        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
    return df

# %% [code] {"papermill":{"duration":3.316613,"end_time":"2024-07-01T02:59:55.727834","exception":false,"start_time":"2024-07-01T02:59:52.411221","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-07-15T20:49:37.121960Z","iopub.execute_input":"2024-07-15T20:49:37.122276Z","iopub.status.idle":"2024-07-15T20:49:41.538280Z","shell.execute_reply.started":"2024-07-15T20:49:37.122250Z","shell.execute_reply":"2024-07-15T20:49:41.537314Z"}}
st = time.time()

# sort by length to optimize dynamic padding
gemma_data = gemma_data.sort_values("length", ascending=False)
# the total #tokens in sub_1 and sub_2 should be more or less the same
gemma_sub_1 = gemma_data.iloc[0::2].copy()
gemma_sub_2 = gemma_data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    gemma_results = executor.map(gemma_inference, (gemma_sub_1, gemma_sub_2), (gemma_model_0, gemma_model_1), (device_0, device_1))

gemma_result_df = pd.concat(list(gemma_results), axis=0)
gemma_result_df = gemma_result_df.sort_index(ascending=True)
gemma_proba = gemma_result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:50:41.439188Z","iopub.execute_input":"2024-07-15T20:50:41.439587Z","iopub.status.idle":"2024-07-15T20:50:41.444310Z","shell.execute_reply.started":"2024-07-15T20:50:41.439543Z","shell.execute_reply":"2024-07-15T20:50:41.443501Z"}}
from numba import cuda
cuda.current_context().memory_manager.deallocations.clear()

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:51:02.083843Z","iopub.execute_input":"2024-07-15T20:51:02.084786Z","iopub.status.idle":"2024-07-15T20:51:20.259839Z","shell.execute_reply.started":"2024-07-15T20:51:02.084749Z","shell.execute_reply":"2024-07-15T20:51:20.258943Z"}}
# delete gemma models
import gc

del gemma_tokenizer
gc.collect()
del gemma_peft_config
gc.collect()
#gemma_base_model_0.cpu()
del gemma_base_model_0
gc.collect()
#gemma_base_model_1.cpu()
del gemma_base_model_1
gc.collect()
#gemma_model_0.to('cpu')
del gemma_model_0
gc.collect()
#gemma_model_1.to('cpu')
del gemma_model_1
gc.collect()
del test
del gemma_data
del gemma_sub_1
del gemma_sub_2
del gemma_result_df
gc.collect()

gc.collect()
with torch.no_grad():
    torch.cuda.set_device('cuda:1')
    torch.cuda.empty_cache()
    torch.cuda.set_device('cuda:0')
    torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:52:33.661383Z","iopub.execute_input":"2024-07-15T20:52:33.662623Z","iopub.status.idle":"2024-07-15T20:52:34.080224Z","shell.execute_reply.started":"2024-07-15T20:52:33.662571Z","shell.execute_reply":"2024-07-15T20:52:34.079120Z"}}
#from numba import cuda
#device = cuda.select_device(0)
#device.reset()

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:54:40.152860Z","iopub.execute_input":"2024-07-15T20:54:40.153245Z","iopub.status.idle":"2024-07-15T20:54:40.158366Z","shell.execute_reply.started":"2024-07-15T20:54:40.153214Z","shell.execute_reply":"2024-07-15T20:54:40.157475Z"}}
print(cuda.list_devices())

# %% [code]
#device = cuda.select_device(1)
#device.reset()

# %% [markdown]
# #### Load Llama3 Model

# %% [code] {"execution":{"iopub.status.busy":"2024-07-15T20:59:26.641219Z","iopub.execute_input":"2024-07-15T20:59:26.641623Z","iopub.status.idle":"2024-07-15T20:59:34.715777Z","shell.execute_reply.started":"2024-07-15T20:59:26.641569Z","shell.execute_reply":"2024-07-15T20:59:34.714400Z"}}
# Load base model on GPU 0
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

MODEL_NAME = '/kaggle/input/llama-3-8b-bnb-4bit'
WEIGHTS_PATH = '/kaggle/input/llama3-unsloth-model-0804-2'
TOKENIZER_PATH = '/kaggle/input/llama3-unsloth-tokenizer-0804-2'
MAX_LENGTH = 1280
BATCH_SIZE = 2
DEVICE = torch.device("cuda")    

test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
def tokenize(example, tokenizer):
    prompt_a = tokenizer('\n\n~~~~~~~~~~ CONVERSATION WITH BOT A ~~~~~~~~~~' + '\n\n<user_prompt>: ' + " ".join(eval(example['prompt'], {"null": ""})), add_special_tokens=False)["input_ids"]
    response_a = tokenizer('\n\n<response_a>: ' + " ".join(eval(example['response_a'], {"null": ""})), add_special_tokens=False)["input_ids"]
    prompt_b = tokenizer('\n\n~~~~~~~~~~ CONVERSATION WITH BOT B ~~~~~~~~~~' + '\n\n<user_prompt>: ' + " ".join(eval(example['prompt'], {"null": ""})), add_special_tokens=False)["input_ids"]
    response_b = tokenizer('\n\n<response_b>: ' + " ".join(eval(example['response_b'], {"null": ""})), add_special_tokens=False)["input_ids"]
    if len(prompt_a+response_a+prompt_b+response_b) > MAX_LENGTH:
        prompt_a = tokenizer('\n\n~~~~~~~~~~ CONVERSATION WITH BOT A ~~~~~~~~~~' + '\n\n<user_prompt>: ' + eval(example['prompt'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:256]
        if len(tokenizer('\n\n<response_a>: ' + eval(example['response_a'], {"null": ""})[-1], add_special_tokens=False)["input_ids"])<=512:
            response_a = tokenizer('\n\n<response_a>: ' + eval(example['response_a'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:512]
        else:
            response_a = tokenizer('\n\n<response_a>: ' + eval(example['response_a'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][-512:]
        prompt_b = tokenizer('\n\n~~~~~~~~~~ CONVERSATION WITH BOT B ~~~~~~~~~~' + '\n\n<user_prompt>: ' + eval(example['prompt'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:256]
        if len(tokenizer('\n\n<response_b>: ' + eval(example['response_b'], {"null": ""})[-1], add_special_tokens=False)["input_ids"])<=512:
            response_b = tokenizer('\n\n<response_b>: ' + eval(example['response_b'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][:512]
        else:
            response_b = tokenizer('\n\n<response_b>: ' + eval(example['response_b'], {"null": ""})[-1], add_special_tokens=False)["input_ids"][-512:]
    start_prompt = tokenizer('Which one of the chatbots below did a better job responding to the user request? BOT A, BOT B or tie?', add_special_tokens=False)["input_ids"]
    end_prompt = tokenizer('\n\n---------\nAnswer: ', add_special_tokens=False)["input_ids"]

    label_token_id = [128250]
    input_ids = [tokenizer.bos_token_id] + start_prompt + prompt_a + response_a + prompt_b + response_b + end_prompt + label_token_id + [tokenizer.eos_token_id]
    attention_mask = len(input_ids)*[1]
    labels = [-100]* len([tokenizer.bos_token_id] + start_prompt + prompt_a + response_a + prompt_b + response_b + end_prompt) + label_token_id + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
LABEL_IDS = [tokenizer(i, add_special_tokens=False)["input_ids"][0] for i in ['a', 'b', 'tie']]
def load_data(df, tokenizer):
    raw_datasets = Dataset.from_pandas(df)
    tokenized_datasets = raw_datasets.map(
        tokenize, 
        # remove_columns=raw_datasets.column_names,
        fn_kwargs={'tokenizer': tokenizer},
    )
    return tokenized_datasets
test_ds = load_data(test, tokenizer)

data = test_ds.to_pandas()
data["max_len"] = data["input_ids"].apply(len)

class Llama3ForSFT(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids = None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states = None,
        return_dict= None,
        cache_position = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            fake_label_tokens_ids = torch.tensor([128250],device=shift_labels.device)
            label_tokens_ids = torch.tensor(LABEL_IDS,device=shift_labels.device)
#             index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
#             true_labels = shift_labels[torch.isin(shift_labels, label_tokens_ids)]
#             true_labels = torch.tensor([index_mapping[label.item()] for label in true_labels], device=true_labels.device)
            true_logits = shift_logits[torch.isin(shift_labels, fake_label_tokens_ids)][:,label_tokens_ids]
#             loss = loss_fct(true_logits, true_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=true_logits,
        )

# Load base model on GPU 0
device0 = torch.device('cuda:0')
base_model_0 = Llama3ForSFT.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    device_map='cuda:0',
#    from_pt=True
)
# Load base model on GPU 1
device1 = torch.device('cuda:1')
base_model_1 = Llama3ForSFT.from_pretrained(
    MODEL_NAME,
    use_cache=False,
    device_map='cuda:1',
#    from_pt=True
)

# Get peft
model_0 = PeftModel.from_pretrained(base_model_0, model_id=WEIGHTS_PATH).to(device0) 
model_0.eval()

model_1 = PeftModel.from_pretrained(base_model_1, model_id=WEIGHTS_PATH).to(device1)
model_1.eval()

@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    a_win, b_win, tie = [], [], []

    model.eval()
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        labels = tmp["labels"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        pad_labels=[]
        for label in labels:
            label = list(label) + [tokenizer.pad_token_id]*(input_ids[0].shape[0]-label.shape[0])
            pad_labels.append(label)
        labels = torch.tensor(pad_labels).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    df['winner_model_a'] = a_win
    df['winner_model_b'] = b_win
    df['winner_tie'] = tie
    return df

st = time.time()

data = data.sort_values("max_len", ascending=False)
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    llama_results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device0, device1))

llama_result_df = pd.concat(list(llama_results), axis=0)
llama_result_df = llama_result_df.sort_index(ascending=True)
llama_proba = llama_result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

# ensemble: simply take average
proba = 0.5*gemma_proba + 0.5*llama_proba

print(f"elapsed time: {time.time() - st}")


#del llama_tokenizer
#gc.collect()
#del llama_peft_config
#gc.collect()
#del llama_base_model_0
#gc.collect()
#del llama_base_model_1
#gc.collect()
#del llama_model_0
#gc.collect()
#del llama_model_1
#gc.collect()
#del llama_data
#del llama_sub_1
#del llama_sub_2
#gc.collect()

#gc.collect()
#with torch.no_grad():
#    torch.cuda.set_device('cuda:1')
#    torch.cuda.empty_cache()
#    torch.cuda.set_device('cuda:0')
#    torch.cuda.empty_cache()


# %% [code] {"papermill":{"duration":0.03061,"end_time":"2024-07-01T02:59:57.532498","exception":false,"start_time":"2024-07-01T02:59:57.501888","status":"completed"},"tags":[]}
result_df = llama_result_df
result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
result_df.loc[:, "winner_tie"] = proba[:, 2]

# no post-processing
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('submission.csv', index=False)
#display(submission_df)

# %% [code]
