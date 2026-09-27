"""
generate.py — Load a trained checkpoint and generate text from a prompt.

Usage:
    python generate.py --prompt "ROMEO:" --tokens 500
"""

import argparse
import torch
from model import TinyGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = TinyGPT(
        cfg["vocab_size"], cfg["n_embd"], cfg["n_head"],
        cfg["n_layer"], cfg["block_size"], cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def encode(s):
        return [stoi[c] for c in s if c in stoi]

    def decode(l):
        return "".join(itos[i] for i in l)

    context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        context, max_new_tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k,
    )[0].tolist()

    print(decode(out))


if __name__ == "__main__":
    main()
