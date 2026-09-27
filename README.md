# Transformer From Scratch

A GPT-style transformer built entirely from first principles in PyTorch —
no `nn.Transformer`, no `nn.MultiheadAttention`. Every piece (self-attention,
multi-head attention, positional embeddings, the transformer block) is
implemented by hand in `model.py`, then trained on Shakespeare's plays at
the character level.

## What this is (and isn't)

**This is a from-scratch implementation of the transformer architecture,
built to understand how models like GPT and Claude work internally** —
not a chatbot and not a coherent Shakespeare generator.

The model learns one task only: given a chunk of characters, predict the
next one. It has no knowledge of Shakespeare's plots, characters, or
meaning — it only learns the *statistical shape* of the text (dialogue
formatting, character-name capitalization, punctuation, rough vocabulary).

At this scale (~10M parameters, ~1MB of training text), the output is
stylistically Shakespeare-like but not semantically coherent — that's an
expected result, not a bug. Real LLMs use billions of parameters and
hundreds of billions of words of training data; this project trades that
scale for full transparency into every line of the architecture.

**Sample output after training** (10.79M params, 5000 steps):
```
KING EDWARD IV:
Thu drump that, for what gave say, he speak at men.

QUEEN:
My blood liegest, leave my cousin;
And I am moiety I in your honourness.
```
Correct dialogue structure and character-name formatting, real English
words, but not coherent meaning — exactly what theory predicts for a
model this size.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows; use source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

For GPU training (NVIDIA only), install the CUDA build of PyTorch instead —
see [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for
the exact command matching your CUDA version.

## Train

```bash
python train.py
```

This trains on `data/input.txt`, prints train/val loss every 300 steps,
and saves a `checkpoint.pt` when done. On a laptop CPU this takes roughly
20-40 minutes for the default config; on a GPU, a few minutes.

To train on your own text, just replace `data/input.txt` with any plain
text file (the more of it, and the more consistent the style, the better
the results).

## Generate text

```bash
python generate.py --prompt "ROMEO:" --tokens 500
```

Flags:
- `--temperature` — higher = more random/creative, lower = more predictable (default 0.8)
- `--top_k` — restricts sampling to the top-k most likely next characters (default 40)

## What you should see while training

- **Loss starts around 4.2** (roughly what random guessing among 65
  characters gives you: `ln(65) ≈ 4.17`)
- **Drops steadily** as the model learns local patterns — spelling,
  punctuation, dialogue formatting
- **Train and validation loss should stay close together.** If train loss
  keeps dropping while validation loss stalls or rises, that's overfitting —
  the model is starting to memorize rather than generalize. In testing here,
  that plateau showed up around step 4500 at this model size.

## Project structure

```
├── model.py           # the transformer architecture, built from scratch
├── train.py           # training loop
├── generate.py        # load a checkpoint and generate text
├── requirements.txt   # exact dependencies
├── data/
│   └── input.txt      # training text (Tiny Shakespeare by default)
└── README.md
```

## Ideas to extend

- Swap the character-level tokenizer for a proper BPE tokenizer
- Add attention-map visualization to see what the model is attending to
- Train on a different corpus (your own notes, song lyrics you own the rights to, etc.)
- Wrap `generate.py` in a small web demo (Flask/FastAPI + a text box)
