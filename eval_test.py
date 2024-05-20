from datasets import load_dataset

dataset = load_dataset("allegro/klej-dyk")

# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#meta-llama-3-instruct

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import numpy as np

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

# Encode the new user input, add the eos_token and return a tensor in Pytorch
inputs = tokenizer.encode("hey", return_tensors='pt').to("cuda")

output = model.generate(
    inputs, 
    max_length=5, 
    pad_token_id=tokenizer.eos_token_id, 
    output_scores=True,
    return_dict_in_generate=True
)

# Get the log probabilities
scores = output.scores

# Get the generated sequence
generated_sequence = output.sequences[0]

# Decode the generated sequence
generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", generated_text)

transition_scores = model.compute_transition_scores(
    output.sequences, output.scores, normalize_logits=True
)

input_length = inputs.shape[1]
generated_tokens = output.sequences[:, input_length:]
total_log_prob = .0
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
    total_log_prob += score

print(f"Trailing log probability: {total_log_prob.cpu().numpy():.4f}")

# logprobs for arbitrary tokens
o = model(input_ids=inputs, labels=inputs, return_dict=True)

logprobs = torch.log_softmax(o.logits, dim=-1)
torch.gather(logprobs, -1, inputs[None,:,:])

def get_logprobs(input_ids, model):
    o = model(input_ids=input_ids, labels=input_ids, return_dict=True)
    logprobs = torch.log_softmax(o.logits, dim=-1)
    return torch.gather(logprobs, -1, input_ids[None,:,:])
