"""
Training script for SmallLM.

Usage:
    python train.py                        # train on built-in sample text
    python train.py --data path/to/file.txt
    python train.py --data input.txt --epochs 20 --batch_size 64
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from data import prepare_data
from model import SmallLM

# --------------------------------------------------------------------------- #
# A tiny public-domain corpus so the model can train with no extra files.
# (First two paragraphs of Frankenstein by Mary Shelley)
# --------------------------------------------------------------------------- #
SAMPLE_TEXT = """
You will rejoice to hear that no disaster has accompanied the commencement
of an enterprise which you have regarded with such evil forebodings. I arrived
here yesterday, and my first task is to assure my dear sister of my welfare
and increasing confidence in the success of my undertaking.

I am already far north of London, and as I walk in the streets of Petersburgh,
I feel a cold northern breeze play upon my cheeks, which braces my nerves and
fills me with delight. Do you understand this feeling? This breeze, which has
travelled from the regions towards which I am advancing, gives me a foretaste
of those icy climes. Inspirited by this wind of promise, my daydreams become
more fervent and vivid. I try in vain to be persuaded that the pole is the
seat of frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight. There, Margaret, the sun is for ever visible, its
broad disk just skirting the horizon and diffusing a perpetual splendour. There—
for with your leave, my sister, I will put some trust in preceding navigators—
there snow and frost are banished; and, sailing over a calm sea, we may be
wafted to a land surpassing in wonders and in beauty every region hitherto
discovered on the habitable globe. Its productions and features may be without
example, as the phenomena of the heavenly bodies undoubtedly are in those
undiscovered solitudes. What may not be expected in a country of eternal light?
I may there discover the wondrous power which attracts the needle and may
regulate a thousand celestial observations that require only this voyage to
render their seeming eccentricities consistent for ever. I shall satiate my
ardent curiosity with the sight of a part of the world never before visited,
and may tread a land never before imprinted by the foot of man. These are my
enticements, and they are sufficient to conquer all fear of danger or death and
to induce me to commence this laborious voyage with the joy a child feels when
he embarks in a little boat, with his holiday mates, on an expedition of
discovery up his native river. But supposing all these conjectures to be false,
you cannot contest the inestimable benefit which I shall confer on all mankind,
to the last generation, by discovering a passage near the pole to those
countries, to reach which at present so many months are requisite; or by
ascertaining the secret of the magnet, which, if at all possible, can only be
effected by an undertaking such as mine.

These reflections have dispelled the agitation with which I began my letter,
and I feel my heart glow with an enthusiasm which elevates me to heaven, for
nothing contributes so much to tranquillise the mind as a steady purpose—a
point on which the soul may fix its intellectual eye. This expedition has been
the favourite dream of my early years. I have read with ardour the accounts of
the various voyages which have been made in the prospect of arriving at the
North Pacific Ocean through the seas which surround the pole. You may remember
that a history of all the voyages made for purposes of discovery composed the
whole of our good Uncle Thomas's library. My education was neglected, yet I was
passionately fond of reading. These volumes were my study day and night, and my
familiarity with them increased that regret which I had felt, as a child, on
learning that my father's dying injunction had forbidden my uncle to allow me to
embark in a seafaring life.

These visions faded when I perused, for the first time, those poets whose
effusions entranced my soul and lifted it to heaven. I also became a poet and
for one year lived in a paradise of my own creation; I imagined that I also
might obtain a niche in the temple where the names of Homer and Shakespeare are
consecrated. You are well acquainted with my failure and how heavily I bore the
disappointment. But just at that time I inherited the fortune of my cousin, and
my thoughts were turned into the channel of their earlier bent.
""".strip()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to a .txt training file")
    p.add_argument("--out", type=str, default="model.pt", help="Where to save the checkpoint")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    text = open(args.data).read() if args.data else SAMPLE_TEXT
    print(f"Corpus size: {len(text):,} characters")

    tokenizer, train_ds, val_ds = prepare_data(text, args.block_size)
    print(f"Vocab size: {tokenizer.vocab_size}  |  "
          f"Train tokens: {len(train_ds):,}  |  Val tokens: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ----------------------------------------------------------------- model
    model = SmallLM(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --------------------------------------------------------------- training
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(),
                 "vocab": tokenizer.vocab,   # plain list — safe to load
                 "config": vars(args)},
                args.out,
            )

    print(f"\nBest val loss: {best_val_loss:.4f}  |  Checkpoint: {args.out}")


if __name__ == "__main__":
    train()
