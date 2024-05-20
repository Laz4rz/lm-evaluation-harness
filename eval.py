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

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)


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
    return torch.gather(logprobs, -1, input_ids[None,:,:])

raise NotImplementedError("This code is not working as expected. It is not clear how to get the logprobs for arbitrary tokens.")

for i, row in example_question_dataset.iterrows():
    question = row["question"]
    answer = row["answer"]
    input_text = original_template.format(question=question, answer=answer)
    value = get_logprobs(input_text, model, tokenizer).sum()
    print(f"Question: {question}\nAnswer: {answer}\Logprobs: {value.item()}")
    true = get_logprobs(input_text + "Tak", model, tokenizer).sum()
    print(get_logprobs(input_text + "Tak", model, tokenizer)[-1])
    false = get_logprobs(input_text + "Nie", model, tokenizer).sum()
    print(f"True: {true.item()}\nFalse: {false.item()}")
    print("Difference true:", value.item() - true.item())
    print("Difference false:", value.item() - false.item())
    print("\n")
