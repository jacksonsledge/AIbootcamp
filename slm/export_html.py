"""
Train a small model and bake it into a self-contained HTML file.
Open the output HTML in any browser — no server required.

Usage:
    python slm/export_html.py                       # train + export
    python slm/export_html.py --data mytext.txt     # custom corpus
    python slm/export_html.py --out demo.html       # custom output path
    python slm/export_html.py --epochs 50           # more training
"""

import argparse
import base64
import json
import struct
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import prepare_data
from model import SmallLM

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
    p.add_argument("--data",       type=str,   default=None)
    p.add_argument("--out",        type=str,   default="slm_demo.html")
    p.add_argument("--block_size", type=int,   default=64)
    p.add_argument("--n_layers",   type=int,   default=3)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--n_embd",     type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def train(args, text):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Corpus: {len(text):,} chars")

    tokenizer, train_ds, val_ds = prepare_data(text, args.block_size)
    print(f"Vocab: {tokenizer.vocab_size}  |  Train tokens: {len(train_ds):,}")

    model = SmallLM(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {model.num_params():,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tloss = sum(
            (lambda loss: (optimizer.zero_grad(), loss.backward(),
                           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                           optimizer.step(), loss.item())[-1])(model(x.to(device), y.to(device))[1])
            for x, y in train_loader
        ) / len(train_loader)

        model.eval()
        with torch.no_grad():
            vloss = sum(model(x.to(device), y.to(device))[1].item() for x, y in val_loader) / max(len(val_loader), 1)

        scheduler.step()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:3d}/{args.epochs}  train={tloss:.4f}  val={vloss:.4f}  ({time.time()-t0:.1f}s)")

    model.load_state_dict(best_state)
    model.cpu().eval()
    print(f"Best val loss: {best_val:.4f}")
    return model, tokenizer


def tensor_to_b64(t: torch.Tensor) -> str:
    arr = t.float().contiguous().numpy()
    raw = struct.pack(f"{arr.size}f", *arr.flat)
    return base64.b64encode(raw).decode()


def export_weights(model: SmallLM) -> dict:
    sd = model.state_dict()
    return {k: tensor_to_b64(v) for k, v in sd.items()}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Small Language Model</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.8rem; font-weight: 700; color: #a78bfa; margin-bottom: 0.3rem; }
  .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
  .card { background: #1e2130; border: 1px solid #2d3148; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem; font-weight: 500; }
  textarea, input[type=text] {
    width: 100%; background: #0f1117; border: 1px solid #2d3148; border-radius: 8px;
    color: #e2e8f0; padding: 0.75rem; font-size: 0.95rem; font-family: inherit;
    resize: vertical; outline: none;
  }
  textarea:focus, input[type=text]:focus { border-color: #7c3aed; }
  .sliders { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .slider-group { display: flex; flex-direction: column; gap: 0.3rem; }
  .slider-row { display: flex; align-items: center; gap: 0.6rem; }
  input[type=range] { flex: 1; accent-color: #7c3aed; }
  .slider-val { font-size: 0.85rem; color: #a78bfa; min-width: 2.5rem; text-align: right; }
  button {
    background: #7c3aed; color: white; border: none; border-radius: 8px;
    padding: 0.7rem 1.8rem; font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: background 0.2s; margin-top: 1rem; width: 100%;
  }
  button:hover { background: #6d28d9; }
  button:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
  #output {
    width: 100%; min-height: 200px; background: #0f1117; border: 1px solid #2d3148;
    border-radius: 8px; padding: 1rem; font-family: 'Courier New', monospace;
    font-size: 0.9rem; color: #86efac; white-space: pre-wrap; word-break: break-word;
    line-height: 1.6;
  }
  .stats { font-size: 0.8rem; color: #64748b; margin-top: 0.5rem; }
  .badge { display: inline-block; background: #1e2130; border: 1px solid #2d3148; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.75rem; margin-right: 0.4rem; color: #94a3b8; }
  .badge span { color: #a78bfa; }
  #status { font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; min-height: 1.2em; }
</style>
</head>
<body>
<div class="container">
  <h1>🧠 Small Language Model</h1>
  <p class="subtitle">A tiny GPT-style transformer running entirely in your browser.</p>

  <div class="card">
    <div id="model-info">
      <span class="badge">Layers <span id="b-layers"></span></span>
      <span class="badge">Heads <span id="b-heads"></span></span>
      <span class="badge">Embed dim <span id="b-embd"></span></span>
      <span class="badge">Context <span id="b-ctx"></span></span>
      <span class="badge">Vocab <span id="b-vocab"></span></span>
      <span class="badge">Params <span id="b-params"></span></span>
    </div>
  </div>

  <div class="card">
    <label for="prompt">Prompt (leave blank to let the model start freely)</label>
    <input type="text" id="prompt" placeholder="e.g.  I feel a cold northern breeze…">

    <div class="sliders">
      <div class="slider-group">
        <label>Tokens to generate</label>
        <div class="slider-row">
          <input type="range" id="tokens" min="20" max="500" step="10" value="200">
          <span class="slider-val" id="tokens-val">200</span>
        </div>
      </div>
      <div class="slider-group">
        <label>Temperature</label>
        <div class="slider-row">
          <input type="range" id="temp" min="0.1" max="2.0" step="0.05" value="0.8">
          <span class="slider-val" id="temp-val">0.8</span>
        </div>
      </div>
      <div class="slider-group">
        <label>Top-k (0 = disabled)</label>
        <div class="slider-row">
          <input type="range" id="topk" min="0" max="100" step="1" value="40">
          <span class="slider-val" id="topk-val">40</span>
        </div>
      </div>
    </div>

    <button id="gen-btn" onclick="startGenerate()">Generate</button>
    <div id="status"></div>
  </div>

  <div class="card">
    <label>Output</label>
    <div id="output"></div>
    <div class="stats" id="gen-stats"></div>
  </div>
</div>

<script>
// ─── model data (injected by export_html.py) ───────────────────────────────
const MODEL_JSON = __MODEL_JSON__;
// ───────────────────────────────────────────────────────────────────────────

// ── Decode base64-encoded float32 weights ──────────────────────────────────
function b64toF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

// ── Tensor math (all flat Float32Arrays, row-major) ────────────────────────

// x: [rows, cols] -> y: [rows, cols], each row normalized
function layernorm(x, rows, cols, w, b) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mean = 0, variance = 0;
    for (let c = 0; c < cols; c++) mean += x[r*cols+c];
    mean /= cols;
    for (let c = 0; c < cols; c++) { const d = x[r*cols+c]-mean; variance += d*d; }
    variance /= cols;
    const std = Math.sqrt(variance + 1e-5);
    for (let c = 0; c < cols; c++)
      out[r*cols+c] = ((x[r*cols+c]-mean)/std) * w[c] + b[c];
  }
  return out;
}

// A:[m,k]  B:[k,n]  ->  C:[m,n]
function matmul(A, B, m, k, n) {
  const C = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      const a = A[i*k+p];
      for (let j = 0; j < n; j++)
        C[i*n+j] += a * B[p*n+j];
    }
  return C;
}

// add bias [n] to each row of [m,n]
function addBias(x, bias, m, n) {
  const out = new Float32Array(x);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      out[i*n+j] += bias[j];
  return out;
}

// residual add: out = a + b
function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

// GELU approximation (tanh form, same as PyTorch)
function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v*v*v)));
  }
  return out;
}

// softmax in-place over last axis of [rows, cols]
function softmax(x, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) if (x[r*cols+c] > max) max = x[r*cols+c];
    let sum = 0;
    for (let c = 0; c < cols; c++) { x[r*cols+c] = Math.exp(x[r*cols+c]-max); sum += x[r*cols+c]; }
    for (let c = 0; c < cols; c++) x[r*cols+c] /= sum;
  }
}

// Multi-head causal self-attention
// x: [T, D], qkv_w: [3D, D] (PyTorch Linear weight transposed),
// proj_w: [D, D]
// returns: [T, D]
function causalAttention(x, T, nHeads, headDim, D, qkvW, projW) {
  // QKV projection: [T, 3D] = x @ qkvW.T
  // qkvW shape is [3D, D] (out_features, in_features)
  // so x @ qkvW.T means: for each t, for each out_j: sum_k x[t,k]*qkvW[j,k]
  const qkv = matmul(x, qkvW, T, D, 3*D); // but qkvW is [3D,D], we want x*qkvW^T -> [T,3D]
  // Actually: qkvW is stored as [out, in] = [3D, D]
  // We need y = x @ W^T, i.e. y[t,j] = sum_k x[t,k]*W[j,k]
  // matmul(x, W^T) where W^T is [D, 3D]... let me just do it directly:
  const QKV = new Float32Array(T * 3 * D);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < 3*D; j++) {
      let s = 0;
      for (let k = 0; k < D; k++) s += x[t*D+k] * qkvW[j*D+k];
      QKV[t*3*D+j] = s;
    }

  // Split Q, K, V: each [T, D]
  const Q = new Float32Array(T * D);
  const K = new Float32Array(T * D);
  const V = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) Q[t*D+d] = QKV[t*3*D+d];
    for (let d = 0; d < D; d++) K[t*D+d] = QKV[t*3*D+D+d];
    for (let d = 0; d < D; d++) V[t*D+d] = QKV[t*3*D+2*D+d];
  }

  const scale = 1.0 / Math.sqrt(headDim);
  const attnOut = new Float32Array(T * D);

  for (let h = 0; h < nHeads; h++) {
    const o = h * headDim;

    // Scores: [T, T] = Q_h @ K_h.T * scale
    const scores = new Float32Array(T * T);
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++) {
        if (j > i) { scores[i*T+j] = -Infinity; continue; } // causal mask
        let s = 0;
        for (let d = 0; d < headDim; d++) s += Q[i*D+o+d] * K[j*D+o+d];
        scores[i*T+j] = s * scale;
      }

    // Softmax each row
    softmax(scores, T, T);

    // Weighted sum of V: [T, headDim]
    for (let i = 0; i < T; i++)
      for (let d = 0; d < headDim; d++) {
        let s = 0;
        for (let j = 0; j < T; j++) s += scores[i*T+j] * V[j*D+o+d];
        attnOut[i*D+o+d] = s;
      }
  }

  // Output projection: [T, D] = attnOut @ projW.T  (projW: [D, D])
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < D; j++) {
      let s = 0;
      for (let k = 0; k < D; k++) s += attnOut[t*D+k] * projW[j*D+k];
      out[t*D+j] = s;
    }
  return out;
}

// Feed-forward: Linear(D,4D) -> GELU -> Linear(4D,D)
// fc1_w: [4D, D], fc1_b: [4D], fc2_w: [D, 4D], fc2_b: [D]
function feedforward(x, T, D, fc1W, fc1B, fc2W, fc2B) {
  const D4 = fc1W.length / D; // 4*D
  // x @ fc1W.T + fc1B: [T, 4D]
  let h = new Float32Array(T * D4);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < D4; j++) {
      let s = fc1B[j];
      for (let k = 0; k < D; k++) s += x[t*D+k] * fc1W[j*D+k];
      h[t*D4+j] = s;
    }
  h = gelu(h);
  // h @ fc2W.T + fc2B: [T, D]
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let j = 0; j < D; j++) {
      let s = fc2B[j];
      for (let k = 0; k < D4; k++) s += h[t*D4+k] * fc2W[j*D4+k];
      out[t*D+j] = s;
    }
  return out;
}

// ── Full forward pass ───────────────────────────────────────────────────────
const cfg  = MODEL_JSON.config;
const W    = {}; // decoded weight cache

function getW(key) {
  if (!W[key]) W[key] = b64toF32(MODEL_JSON.weights[key]);
  return W[key];
}

function modelForward(ids) {
  const T = ids.length;
  const D = cfg.n_embd;

  // Token + position embeddings
  const tokEmb = getW("tok_emb.weight");  // [V, D]
  const posEmb = getW("pos_emb.weight");  // [block_size, D]

  let x = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let d = 0; d < D; d++)
      x[t*D+d] = tokEmb[ids[t]*D+d] + posEmb[t*D+d];

  // Transformer blocks
  for (let l = 0; l < cfg.n_layers; l++) {
    const prefix = `blocks.${l}`;
    const ln1W = getW(`${prefix}.ln1.weight`);
    const ln1B = getW(`${prefix}.ln1.bias`);
    const ln2W = getW(`${prefix}.ln2.weight`);
    const ln2B = getW(`${prefix}.ln2.bias`);
    const qkvW = getW(`${prefix}.attn.qkv.weight`);   // [3D, D]
    const projW= getW(`${prefix}.attn.proj.weight`);  // [D, D]
    const fc1W = getW(`${prefix}.ff.net.0.weight`);   // [4D, D]
    const fc1B = getW(`${prefix}.ff.net.0.bias`);
    const fc2W = getW(`${prefix}.ff.net.2.weight`);   // [D, 4D]
    const fc2B = getW(`${prefix}.ff.net.2.bias`);

    const nHeads  = cfg.n_heads;
    const headDim = D / nHeads;

    // Pre-norm attention + residual
    const xn1  = layernorm(x, T, D, ln1W, ln1B);
    const attn = causalAttention(xn1, T, nHeads, headDim, D, qkvW, projW);
    x = add(x, attn);

    // Pre-norm feedforward + residual
    const xn2 = layernorm(x, T, D, ln2W, ln2B);
    const ff  = feedforward(xn2, T, D, fc1W, fc1B, fc2W, fc2B);
    x = add(x, ff);
  }

  // Final layernorm
  const lnFW = getW("ln_f.weight");
  const lnFB = getW("ln_f.bias");
  x = layernorm(x, T, D, lnFW, lnFB);

  // Project to vocab (weight-tied: use tok_emb.weight as lm_head)
  // logits[last] = x[last] @ tokEmb.T  -> [V]
  const tokEmb2 = getW("tok_emb.weight");  // [V, D]
  const V = cfg.vocab_size;
  const logits = new Float32Array(V);
  const lastX  = x.subarray((T-1)*D, T*D);
  for (let v = 0; v < V; v++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += lastX[d] * tokEmb2[v*D+d];
    logits[v] = s;
  }
  return logits;
}

// ── Sampling ────────────────────────────────────────────────────────────────
function sample(logits, temperature, topK) {
  // Scale by temperature
  const V = logits.length;
  const scaled = new Float32Array(V);
  for (let i = 0; i < V; i++) scaled[i] = logits[i] / temperature;

  // Top-k masking
  if (topK > 0 && topK < V) {
    const sorted = Array.from(scaled).map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v);
    const threshold = sorted[topK-1].v;
    for (let i = 0; i < V; i++) if (scaled[i] < threshold) scaled[i] = -Infinity;
  }

  // Softmax
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (scaled[i] > max) max = scaled[i];
  let sum = 0;
  const probs = new Float32Array(V);
  for (let i = 0; i < V; i++) { probs[i] = Math.exp(scaled[i]-max); sum += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= sum;

  // Multinomial sample
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < V; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return V - 1;
}

// ── UI logic ────────────────────────────────────────────────────────────────
const vocab   = MODEL_JSON.vocab;
const c2i     = {};
vocab.forEach((c,i) => c2i[c] = i);

function encode(text) { return text.split("").map(c => c2i[c] !== undefined ? c2i[c] : c2i[" "]).filter(x => x !== undefined); }
function decode(ids)  { return ids.map(i => vocab[i]).join(""); }

// Fill model info badges
document.getElementById("b-layers").textContent = cfg.n_layers;
document.getElementById("b-heads").textContent  = cfg.n_heads;
document.getElementById("b-embd").textContent   = cfg.n_embd;
document.getElementById("b-ctx").textContent    = cfg.block_size;
document.getElementById("b-vocab").textContent  = cfg.vocab_size;
document.getElementById("b-params").textContent = MODEL_JSON.num_params.toLocaleString();

// Wire up sliders
for (const id of ["tokens","temp","topk"]) {
  const el = document.getElementById(id);
  const lbl = document.getElementById(id+"-val");
  lbl.textContent = el.value;
  el.addEventListener("input", () => lbl.textContent = el.value);
}

let generating = false;

function startGenerate() {
  if (generating) return;
  generating = true;
  const btn = document.getElementById("gen-btn");
  btn.disabled = true;
  btn.textContent = "Generating…";

  const promptText = document.getElementById("prompt").value || " ";
  const maxTokens  = parseInt(document.getElementById("tokens").value);
  const temp       = parseFloat(document.getElementById("temp").value);
  const topK       = parseInt(document.getElementById("topk").value);

  document.getElementById("output").textContent = "";
  document.getElementById("gen-stats").textContent = "";
  document.getElementById("status").textContent = "Running…";

  // Use setTimeout to let the UI update before the heavy compute
  setTimeout(() => {
    try {
      const t0 = performance.now();
      let ids = encode(promptText);
      if (ids.length === 0) ids = [0];

      const outEl = document.getElementById("output");
      const generated = [];

      for (let step = 0; step < maxTokens; step++) {
        const ctx  = ids.slice(-cfg.block_size);
        const logits = modelForward(ctx);
        const next = sample(logits, temp, topK);
        ids.push(next);
        generated.push(next);
      }

      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      const tokPerSec = (maxTokens / elapsed).toFixed(1);
      outEl.textContent = decode(ids);
      document.getElementById("gen-stats").textContent =
        `${maxTokens} tokens in ${elapsed}s (${tokPerSec} tok/s)`;
      document.getElementById("status").textContent = "Done.";
    } catch(e) {
      document.getElementById("status").textContent = "Error: " + e.message;
    } finally {
      generating = false;
      btn.disabled = false;
      btn.textContent = "Generate";
    }
  }, 10);
}
</script>
</body>
</html>
"""


def build_html(model, tokenizer, config):
    sd = model.state_dict()
    weights = {k: tensor_to_b64(v) for k, v in sd.items()}

    model_json = {
        "config": config,
        "vocab": tokenizer.vocab,
        "num_params": model.num_params(),
        "weights": weights,
    }

    json_str = json.dumps(model_json, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__MODEL_JSON__", json_str)


def main():
    args = get_args()
    text = open(args.data).read() if args.data else SAMPLE_TEXT

    print("Training…")
    model, tokenizer = train(args, text)

    config = {
        "vocab_size":  tokenizer.vocab_size,
        "block_size":  args.block_size,
        "n_layers":    args.n_layers,
        "n_heads":     args.n_heads,
        "n_embd":      args.n_embd,
    }

    print(f"\nExporting to {args.out}…")
    html = build_html(model, tokenizer, config)
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = len(html.encode()) / 1024
    print(f"Done. {out_path} ({size_kb:.0f} KB)")
    print(f"Open {out_path} in your browser to interact with the model.")


if __name__ == "__main__":
    main()
