"""
Gradio front-end for SmallLM — creates a public shareable URL automatically.

Run:
    python slm/gradio_app.py
"""

import time
import torch
from torch.utils.data import DataLoader
import gradio as gr

from model import SmallLM
from data import prepare_data

# ------------------------------------------------------------------ defaults

DEFAULT_TEXT = """
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
broad disk just skirting the horizon and diffusing a perpetual splendour.
There snow and frost are banished; and, sailing over a calm sea, we may be
wafted to a land surpassing in wonders and in beauty every region hitherto
discovered on the habitable globe. What may not be expected in a country of
eternal light? I may there discover the wondrous power which attracts the
needle and may regulate a thousand celestial observations. I shall satiate my
ardent curiosity with the sight of a part of the world never before visited,
and may tread a land never before imprinted by the foot of man.

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
""".strip()

# Global model state
_state = {"model": None, "tokenizer": None, "block_size": 128}


# ------------------------------------------------------------------ training

def train_model(
    corpus, epochs, batch_size, lr,
    block_size, n_layers, n_heads, n_embd, dropout,
    progress=gr.Progress()
):
    if len(corpus.strip()) < 200:
        raise gr.Error("Corpus is too short — paste at least 200 characters of text.")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_ds, val_ds = prepare_data(corpus, block_size)

    if len(train_ds) < batch_size:
        raise gr.Error(
            f"Not enough tokens for batch_size={batch_size} with block_size={block_size}. "
            "Use more text or reduce batch/block size."
        )

    model = SmallLM(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads,
        n_embd=n_embd,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    best_val, best_state = float("inf"), None
    log_lines = [
        f"Device: {device.upper()}  |  Vocab: {tokenizer.vocab_size}  |  "
        f"Params: {model.num_params():,}  |  Train tokens: {len(train_ds):,}\n"
        + "-" * 60
    ]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                v_total += loss.item()
        val_loss = v_total / max(len(val_loader), 1)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        log_lines.append(
            f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
            f"val={val_loss:.4f}  ({elapsed:.1f}s)"
        )
        progress(epoch / epochs, desc=f"Epoch {epoch}/{epochs} — val loss {val_loss:.4f}")
        yield "\n".join(log_lines), gr.update(interactive=False)

    model.load_state_dict(best_state)
    _state["model"] = model.cpu()
    _state["tokenizer"] = tokenizer
    _state["block_size"] = block_size

    log_lines.append("-" * 60)
    log_lines.append(f"Done. Best val loss: {best_val:.4f}  — ready to generate!")
    yield "\n".join(log_lines), gr.update(interactive=True)


# ----------------------------------------------------------------- generation

def generate_text(prompt, max_tokens, temperature, top_k):
    if _state["model"] is None:
        raise gr.Error("Train a model first!")

    model     = _state["model"]
    tokenizer = _state["tokenizer"]
    ctx       = _state["block_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    safe = "".join(c if c in tokenizer._c2i else " " for c in (prompt or " "))
    ids  = torch.tensor(tokenizer.encode(safe), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k > 0 else None,
        )

    return tokenizer.decode(out[0].tolist())


# ----------------------------------------------------------------- UI layout

with gr.Blocks(title="Small Language Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Small Language Model\nA tiny GPT-style transformer you can train and sample right in the browser.")

    with gr.Tabs():
        # ---- Train tab ----
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column(scale=2):
                    corpus_box = gr.Textbox(
                        label="Training corpus",
                        value=DEFAULT_TEXT,
                        lines=14,
                        placeholder="Paste any text here (the more the better)…",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Model architecture**")
                    block_size = gr.Slider(32, 256, value=128, step=32, label="Context length")
                    n_layers   = gr.Slider(1, 8,   value=4,          label="Transformer layers")
                    n_heads    = gr.Slider(1, 8,   value=4,          label="Attention heads")
                    n_embd     = gr.Radio([32, 64, 128, 256], value=128, label="Embedding dim")
                    dropout    = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Dropout")
                    gr.Markdown("**Training**")
                    epochs     = gr.Slider(5, 200,  value=30,  step=5,  label="Epochs")
                    batch_size = gr.Slider(8, 128,  value=32,  step=8,  label="Batch size")
                    lr         = gr.Radio([1e-4, 3e-4, 1e-3], value=3e-4, label="Learning rate")

            train_btn  = gr.Button("Train model", variant="primary")
            train_log  = gr.Textbox(label="Training log", lines=12, interactive=False)

        # ---- Generate tab ----
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_box  = gr.Textbox(label="Prompt", placeholder="Leave blank for a free start…")
                    max_tokens  = gr.Slider(50, 1000, value=300, step=50, label="Tokens to generate")
                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05,
                                           label="Temperature (higher = more creative)")
                    top_k       = gr.Slider(0, 100, value=40, step=5,
                                           label="Top-k (0 = disabled)")
                    gen_btn     = gr.Button("Generate", variant="primary", interactive=False)

                with gr.Column(scale=2):
                    output_box = gr.Textbox(label="Generated text", lines=20, interactive=False)

    # ---- wire up events ----
    train_btn.click(
        fn=train_model,
        inputs=[corpus_box, epochs, batch_size, lr,
                block_size, n_layers, n_heads, n_embd, dropout],
        outputs=[train_log, gen_btn],
    )

    gen_btn.click(
        fn=generate_text,
        inputs=[prompt_box, max_tokens, temperature, top_k],
        outputs=[output_box],
    )


if __name__ == "__main__":
    demo.launch(share=True)
