from fastapi import FastAPI
from pydantic import BaseModel
from dataset import tokenizer
from training import generate_text
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    prompt: str


@app.post("/chat")
async def chat(request: PromptRequest):

    token_ids = tokenizer.encode(request.prompt, allowed_special={'<|endoftext|>'})
    token_ids = torch.tensor(token_ids).unsqueeze(0)

    prompt_len = token_ids.shape[1]

    output_ids = generate_text(token_ids, max_num_tokens=50, eos_id=50256, top_k=25, temperature=1.4)
    generated_ids = output_ids[:, prompt_len:]

    decoded_text = tokenizer.decode(generated_ids[0].tolist())

    return {"response": decoded_text}
