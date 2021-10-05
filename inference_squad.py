import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_from_disk

import os
import numpy as np
from tqdm import tqdm

    import torch_spotlight
from torch_spotlight.utils import *

# Note: we load cached copies of the dataset, tokenizer, and model to make inference work without an internet connection
data_dir = os.environ['DATA_DIR'] 
squad_dir = os.path.join(data_dir, 'squad')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'squad')

# Helper function: apply tokenizer to dataset
def prepare_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# Load tokenizer + model
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-distilled-squad", 
    cache_dir=model_path, 
    local_files_only=True
)

model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad", 
    cache_dir=model_path,
    local_files_only=True
)
model.eval()
model.to('cuda')

# Load validation set
print('Loading dataset...')
dataset = load_from_disk(squad_dir)['validation']

def filter_short_examples(example):
    example_length = len(tokenizer(
        example["question"],
        example["context"],
    )['input_ids'])
    return example_length < 384
short_dataset = dataset.filter(filter_short_examples)
features = short_dataset.map(prepare_features, batched=True, remove_columns=dataset.column_names)


# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook

hook_handle = model.qa_outputs.register_forward_hook(get_input('last_layer'))

# Run inference on entire dataset
print('Running inference...')
hidden_list = []
loss_list = []
with torch.no_grad():
    for i in tqdm(range(len(features))):
        batch = {k: torch.tensor(v).cuda() for k, v in features[i:i+1].items()}
        output = model(**batch)
        loss = output.loss.cpu().detach().item()
        
        hidden_list.append(hidden_layers['last_layer'].squeeze().flatten(start_dim=1).cpu())
        loss_list.append(loss)

labels = torch.Tensor(label_list)
embeddings = torch.stack(hidden_list)
outputs = torch.stack(output_list)
losses = torch.Tensor(loss_list)
        

# note: [CLS] token is always first token in this dataset
# can confirm with:

# tokens_list = []
# for i in tqdm(range(len(features))):
#     tokens_list.append(torch.tensor(features[i:i+1]['input_ids']).flatten())
# tokens = torch.stack(tokens_list)
# cls_token_positions = (tokens == tokenizer.cls_token_id).sum(axis=0)
# cls_token_positions[1:].any()

embeddings_cls = embeddings[:, 0, :]
results_cls = InferenceResults(
    embeddings = torch.clone(embeddings_cls),
    outputs    = outputs,
    losses     = losses,
    labels     = labels,
)

saveResults(
    'inference_results/squad_val_bert.pkl',
    results_cls
)

