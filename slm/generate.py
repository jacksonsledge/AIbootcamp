"""
Generate text from a trained SmallLM checkpoint.

Usage:
    python generate.py                              # loads model.pt, random seed
    python generate.py --prompt "The sun"           # continue from a prompt
    python generate.py --ckpt model.pt --tokens 200 --temperature 0.8 --top_k 40
"""

import argparse

import torch

from model import SmallLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="model.pt")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--tokens", type=int, default=300, help="Tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (lower = more predictable)")
    p.add_argument("--top_k", type=int, default=40,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def generate():
    args = get_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    # Reconstruct tokenizer from saved vocab list
    from data import CharTokenizer
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.vocab = ckpt["vocab"]
    tokenizer.vocab_size = len(tokenizer.vocab)
    tokenizer._c2i = {c: i for i, c in enumerate(tokenizer.vocab)}
    tokenizer._i2c = {i: c for i, c in enumerate(tokenizer.vocab)}

    model = SmallLM(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg["block_size"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_embd=cfg["n_embd"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = args.prompt or " "
    # Silently replace unknown characters with space
    prompt = "".join(c if c in tokenizer._c2i else " " for c in prompt)

    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    top_k = args.top_k if args.top_k > 0 else None

    with torch.no_grad():
        out_ids = model.generate(ids, args.tokens, temperature=args.temperature, top_k=top_k)

    generated = tokenizer.decode(out_ids[0].tolist())
    print(generated)


if __name__ == "__main__":
    generate()
