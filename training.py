import torch
from dataset import train_dataloader, test_dataloader, tokenizer
from gpt2code import model


def generate_text(inputs, max_num_tokens, eos_id, top_k=None, temperature=0.0):

    for i in range(max_num_tokens):

        with torch.inference_mode():
            logits = model(inputs)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, top_k)
                min_value = top_k_logits[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, torch.tensor(float('-inf')), logits)

            if temperature > 0.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
        
            next_token_id = torch.multinomial(probs, num_samples=1)

            eos_mask = (next_token_id == eos_id)

            if eos_mask.all():
                break

            inputs = torch.cat((inputs, next_token_id), dim=1)

    return inputs


# batch = next(iter(train_dataloader))

# inputs, targets = batch

# prompt_len = inputs.shape[1]

# output_ids = generate_text(inputs, max_num_tokens=20, top_k=25, temperature=1.4, eos_id=50256)

# generated_ids = output_ids[:, prompt_len:]

# decoded_txt = tokenizer.decode(generated_ids[0].tolist())

# print(decoded_txt)