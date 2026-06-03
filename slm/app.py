"""
Streamlit front-end for SmallLM.

Run:
    streamlit run slm/app.py
"""

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import streamlit as st

from model import SmallLM
from data import CharTokenizer, TextDataset, prepare_data

# ------------------------------------------------------------------ constants

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
and may tread a land never before imprinted by the foot of man.
""".strip()

# ------------------------------------------------------------------ page setup

st.set_page_config(page_title="Small Language Model", page_icon="🧠", layout="wide")
st.title("🧠 Small Language Model")
st.caption("A tiny GPT-style transformer you can train and sample right in the browser.")

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Model config")
    block_size = st.slider("Context length (tokens)", 32, 256, 128, step=32)
    n_layers   = st.slider("Transformer layers",       1, 8,   4)
    n_heads    = st.slider("Attention heads",           1, 8,   4)
    n_embd     = st.select_slider("Embedding dim", options=[32, 64, 128, 256], value=128)
    dropout    = st.slider("Dropout", 0.0, 0.5, 0.1, step=0.05)

    st.divider()
    st.header("Training config")
    epochs     = st.slider("Epochs",      1, 200, 30)
    batch_size = st.slider("Batch size",  8, 128, 32, step=8)
    lr         = st.select_slider("Learning rate",
                                  options=[1e-4, 3e-4, 1e-3, 3e-3], value=3e-4,
                                  format_func=lambda v: f"{v:.0e}")
    seed       = st.number_input("Random seed", value=42, step=1)

# ------------------------------------------------------------------ tabs

tab_train, tab_generate = st.tabs(["Train", "Generate"])

# ===========================================================================
# TRAIN TAB
# ===========================================================================
with tab_train:
    st.subheader("Training corpus")

    upload = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    if upload is not None:
        corpus = upload.read().decode("utf-8", errors="replace")
        st.success(f"Loaded {len(corpus):,} characters from {upload.name}")
    else:
        corpus = st.text_area(
            "Or paste / edit text directly",
            value=DEFAULT_TEXT,
            height=260,
        )
        st.caption(f"{len(corpus):,} characters")

    if st.button("Train model", type="primary", disabled=len(corpus) < 100):
        torch.manual_seed(int(seed))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer, train_ds, val_ds = prepare_data(corpus, block_size)

        if len(train_ds) < batch_size:
            st.error(
                f"Corpus too short for batch_size={batch_size} and "
                f"block_size={block_size}. Add more text or reduce these values."
            )
            st.stop()

        model = SmallLM(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            n_embd=n_embd,
            dropout=dropout,
        ).to(device)

        col1, col2, col3 = st.columns(3)
        col1.metric("Parameters", f"{model.num_params():,}")
        col2.metric("Vocab size", tokenizer.vocab_size)
        col3.metric("Device", device.upper())

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        progress_bar   = st.progress(0, text="Starting training…")
        loss_chart_ph  = st.empty()
        status_ph      = st.empty()

        train_losses, val_losses = [], []
        best_val = float("inf")
        best_state = None

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

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            frac = epoch / epochs
            progress_bar.progress(frac, text=f"Epoch {epoch}/{epochs} — train {train_loss:.4f} | val {val_loss:.4f}")

            # Update chart every 5 epochs to avoid flickering
            if epoch % 5 == 0 or epoch == epochs:
                import pandas as pd
                df = pd.DataFrame({"train": train_losses, "val": val_losses},
                                  index=range(1, epoch + 1))
                df.index.name = "epoch"
                loss_chart_ph.line_chart(df)

        model.load_state_dict(best_state)

        # Persist to session state so Generate tab can use it
        st.session_state["model"]     = model.cpu()
        st.session_state["tokenizer"] = tokenizer
        st.session_state["block_size"] = block_size

        progress_bar.empty()
        status_ph.success(f"Training complete — best val loss: {best_val:.4f}")

# ===========================================================================
# GENERATE TAB
# ===========================================================================
with tab_generate:
    if "model" not in st.session_state:
        st.info("Train a model first, then come back here to generate text.")
    else:
        model     = st.session_state["model"]
        tokenizer = st.session_state["tokenizer"]
        ctx       = st.session_state["block_size"]

        st.subheader("Generate text")

        prompt      = st.text_input("Prompt (leave blank to let the model start freely)", value="")
        max_tokens  = st.slider("Tokens to generate", 50, 1000, 300, step=50)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, step=0.05,
                                help="Higher = more random, lower = more predictable")
        top_k       = st.slider("Top-k sampling", 0, 100, 40,
                                help="0 = disabled (pure sampling)")
        gen_seed    = st.number_input("Generation seed (0 = random)", value=0, step=1)

        if st.button("Generate", type="primary"):
            if int(gen_seed) != 0:
                torch.manual_seed(int(gen_seed))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()

            safe_prompt = "".join(c if c in tokenizer._c2i else " " for c in prompt) or " "
            ids = torch.tensor(tokenizer.encode(safe_prompt), dtype=torch.long, device=device).unsqueeze(0)

            with st.spinner("Generating…"):
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None,
                    )

            generated = tokenizer.decode(out[0].tolist())
            st.text_area("Output", value=generated, height=400)

            # Show a few stats
            col1, col2 = st.columns(2)
            col1.metric("Characters generated", len(generated))
            col2.metric("Vocab size", tokenizer.vocab_size)
