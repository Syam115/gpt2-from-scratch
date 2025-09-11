import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


tokenizer = tiktoken.get_encoding('gpt2')

with open('book.txt', 'r') as file:
    data = file.read()


class GPTDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

split_size = int(0.8 * len(data))
train_data = data[:split_size]
test_data = data[split_size:]

def create_dataloader(txt):
    dataset = GPTDataset(txt, max_length=256, stride=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    return dataloader


train_dataloader = create_dataloader(train_data)
test_dataloader = create_dataloader(test_data)