"""
train.py — Trains TinyGPT on a text file, character by character.

Usage:
    python train.py

Put your training text at data/input.txt (any plain text file — the
bigger and more consistent in style, the better it'll learn).
"""

import torch
from model import TinyGPT

# ---------------- Hyperparameters ----------------
batch_size = 64
block_size = 256          # was 128 → sees twice as much context per prediction
max_iters = 5000          # was 3000 → more training steps
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384               # was 192 → bigger internal representation
n_head = 6
n_layer = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
# ---------------------------------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# --- build character-level vocabulary ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join(itos[i] for i in l)


# --- train / val split ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params/1e6:.2f}M parameters, vocab size {vocab_size}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- save checkpoint + vocab so generate.py can reload everything ---
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "block_size": block_size,
                "dropout": dropout,
            },
        },
        "checkpoint.pt",
    )
    print("Saved checkpoint.pt")

    # quick sample after training
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=300)[0].tolist()
    print("\n--- sample generation ---")
    print(decode(sample))


if __name__ == "__main__":
    main()
