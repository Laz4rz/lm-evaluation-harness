from datasets import load_dataset

dataset = load_dataset("allegro/klej-dyk")

original_template = "Pytanie: \"{question}\"\nSugerowana odpowiedź: \"{answer}\"\nPytanie: Czy sugerowana odpowiedź na zadane pytanie jest poprawna?\nOdpowiedz krótko \"Tak\" lub \"Nie\". Prawidłowa odpowiedź:"
example_question_dataset = dataset.filter(lambda x: x["q_id"] == "czywiesz445")["test"].to_pandas()

# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#meta-llama-3-instruct

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import numpy as np

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {model_id} to device {device}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
print("Model loaded.")

def get_loss_of_arbitrary_tokens(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(device)
    return model(
        input_ids=input_ids.input_ids, 
        labels=input_ids.input_ids, 
        attention_mask=input_ids.attention_mask,
        return_dict=True).loss

def get_logprobs(input_text, model, tokenizer):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    o = model(input_ids=input_ids, labels=input_ids, return_dict=True)
    logprobs = torch.log_softmax(o.logits, dim=-1)
    return torch.gather(logprobs[:,:-1,:], 2, input_ids.unsqueeze(-1)[:, 1:, :]).squeeze(-1)

for i, row in example_question_dataset.iterrows():
    question = row["question"]
    answer = row["answer"]
    input_text = original_template.format(question=question, answer=answer)
    value = get_logprobs(input_text, model, tokenizer).sum()
    print(f"Question: {question}\nAnswer: {answer}\nLogprobs: {value.item()}\nLoss: {get_loss_of_arbitrary_tokens(input_text, model, tokenizer)}\n")
    
    tak_sen_val = get_logprobs(input_text + " Tak", model, tokenizer).sum()
    tak_val = get_logprobs(input_text + " Tak", model, tokenizer)[:,-1].item()
    nie_sen_val = get_logprobs(input_text + " Nie", model, tokenizer).sum()
    nie_val = get_logprobs(input_text + " Nie", model, tokenizer)[:,-1].item()

    print(f"Tak sentence: {tak_sen_val.item()}\nTak: {tak_val}\nNie sentence: {nie_sen_val.item()}\nNie: {nie_val}")

    print("\n")

input_str = input_text + " Nie"
input_tok = tokenizer(input_str, return_tensors="pt")
input_ids = input_tok.input_ids.to(device)
o = model(input_ids, labels=input_ids, return_dict=True)
logprobs = torch.log_softmax(o.logits, dim=-1)
