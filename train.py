import torch

# -----------------------------------------------------------------------------
# 1. Hyperparameters
# -----------------------------------------------------------------------------
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# 2. Data Loading & Tokenizer (Boilerplate)
# -----------------------------------------------------------------------------
# Read the dataset
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers (Character-level Tokenizer)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of ints -> string

# Create Train and Validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% train, 10% val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Generates a small batch of data of inputs (x) and targets (y)"""
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """A helper function to calculate loss on train/val sets without updating weights"""
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set back to training mode
    return out

# -----------------------------------------------------------------------------
# 3. Model Initialization & Training Loop 
# -----------------------------------------------------------------------------
# IMPORTANT: Keep this section commented out until you build model.py!

# from model import GPTLanguageModel
# 
# model = GPTLanguageModel(vocab_size)
# model = model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 
# for iter in range(max_iters):
#     # Every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss(model)
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
# 
#     # sample a batch of data
#     xb, yb = get_batch('train')
# 
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# print("Final training loss:", loss.item())