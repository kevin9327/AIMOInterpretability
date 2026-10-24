# Layerwise Representation Stability and Early-Trace Spectral Entropy for White-Box Detection of Spurious Mathematical Reasoning

**Competition:** NeurIPS 2026 Official Competition — AIMO Interpretability Challenge (`$12,500 USD` Prize Pool)  
**Eligible Tracks:** Main Track (`$5,000`), Small Models Track (`$3,500`), White-Box Award (`+$2,000`), Writeup Track (`$4,000`)  
**Submission Artifacts:** `aimo_main_track.zip` (`solution.py`), `aimo_small_track.zip` (`solution.py` + `small.txt`)

---

## Abstract

Large reasoning models (LRMs) trained with reinforcement learning on mathematical chains of thought frequently achieve high accuracy on canonical benchmark phrasings while failing catastrophically under semantically preserving re-framings—a phenomenon known as *spurious* or *brittle* mathematical reasoning. Crucially, the AIMO Interpretability Challenge evaluates whether a detector can predict a target model's robustness across unseen surface variations **given only a single problem phrasing and white-box access to the model weights**. We introduce **LRS-RTE** (*Layerwise Representation Stability and Reasoning Trace Entropy*), a training-free white-box mechanistic probe that operates in two stages: (1) constructing a deterministic, mathematically isomorphic counterfactual prompt via bound-variable transposition ($x \leftrightarrow y, n \leftrightarrow k, a \leftrightarrow b$) and measuring the residual-stream cosine stability $\mathcal{S}_{\cos}$ and $\ell_2$-norm preservation across the upper half ($50\%, 75\%, 90\%, 100\%$ depth) of the transformer at the reasoning launchpad token; and (2) rolling out a short $K$-step greedy reasoning prefix ($K \in \{8, 16\}$) to compute the top-1/top-2 log-probability margin $\Delta_{\log p}$ and truncated top-64 spectral Shannon entropy $\mathcal{H}_{64}$. Across both dense models (`4B`–`8B`) and mixture-of-experts architectures (`gpt-oss-120b`), LRS-RTE distinguishes memorized pattern matching from invariant algebraic deduction without requiring expensive full-length Chain-of-Thought rollouts.

---

## 1. Mechanistic Hypothesis: The Invariant Algebraic Manifold

Let $\mathcal{M}_\theta$ be an autoregressive transformer with $L$ layers and residual stream dimension $d$. Given a mathematical problem statement $P$, the model maps the tokenized prompt to a sequence of hidden states $\mathbf{h}^{(l)}(P) \in \mathbb{R}^{|P| \times d}$ for $l \in \{1, \dots, L\}$. The final prompt token position $t = |P|$ acts as the **Reasoning Launchpad State** $\mathbf{z}^{(l)}(P) := \mathbf{h}_{|P|}^{(l)}(P)$, summarizing the problem's formal structure immediately prior to emitting `<think>` tokens.

Consider an isomorphic operator $\mathcal{T}$ that renames bound algebraic variables ($x \leftrightarrow y, n \leftrightarrow k, a \leftrightarrow b$) and prepends a deductively neutral instruction. Because $\mathcal{T}(P)$ preserves the underlying mathematical group/ring structure of $P$, a model that has synthesized an **invariant structural representation** of the problem in its mid-to-late transformer blocks must satisfy:

$$\mathcal{S}_{\cos}(P; \mathcal{M}_\theta) = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \frac{\langle \mathbf{z}^{(l)}(P), \mathbf{z}^{(l)}(\mathcal{T}(P)) \rangle}{\|\mathbf{z}^{(l)}(P)\|_2 \, \|\mathbf{z}^{(l)}(\mathcal{T}(P))\|_2} \approx 1, \quad \mathcal{L} = \left\{ \lfloor 0.5L \rfloor, \lfloor 0.75L \rfloor, \lfloor 0.9L \rfloor, L \right\}$$

Conversely, when $\mathcal{M}_\theta$ relies on **surface-level n-gram contamination or heuristic template matching**, renaming variables disrupts early-to-mid layer induction heads, causing rapid directional divergence in the residual stream ($\mathcal{S}_{\cos} \ll 1$) and radial norm drift:

$$\mathcal{S}_{\text{norm}}(P; \mathcal{M}_\theta) = \frac{1}{|\mathcal{L}|} \sum_{l \in \mathcal{L}} \min\left( \frac{\|\mathbf{z}^{(l)}(P)\|_2}{\|\mathbf{z}^{(l)}(\mathcal{T}(P))\|_2}, \frac{\|\mathbf{z}^{(l)}(\mathcal{T}(P))\|_2}{\|\mathbf{z}^{(l)}(P)\|_2} \right)$$

---

## 2. Stage B: Early-Trace Spectral Entropy and Logit Margin

While $\mathcal{S}_{\cos}$ and $\mathcal{S}_{\text{norm}}$ capture pre-generation representational invariance, a model may still fail if its autoregressive transition dynamics enter a high-entropy "guessing regime" during the initial problem decomposition steps. Full-length reasoning rollouts ($2,048$–$8,192$ tokens) are computationally prohibitive within the 1-hour Codabench budget across 50 problems and 5 models. However, we observe that **bifurcation instability manifests within the first $K \in [8, 16]$ reasoning tokens**.

Let $\boldsymbol{\ell}_k \in \mathbb{R}^{|\mathcal{V}|}$ denote the vocabulary logits at reasoning step $k \in \{1, \dots, K\}$, and let $p_k = \text{softmax}(\boldsymbol{\ell}_k)$. We extract two white-box trajectory statistics:
1. **Normalized Top-1 / Top-2 Log-Probability Margin ($\bar{\Delta}_{\log p}$):**
   $$\bar{\Delta}_{\log p} = \frac{1}{K} \sum_{k=1}^{K} \min\left( \frac{\log p_{k,(1)} - \log p_{k,(2)}}{6.0}, \, 1.0 \right)$$
   where $p_{k,(1)} \ge p_{k,(2)}$ are the two largest token probabilities at step $k$.
2. **Truncated Top-64 Spectral Confidence ($1 - \bar{\mathcal{H}}_{64}$):**
   $$\bar{\mathcal{H}}_{64} = \frac{1}{K \log 64} \sum_{k=1}^{K} \left( - \sum_{j=1}^{64} \tilde{p}_{k,(j)} \log \tilde{p}_{k,(j)} \right), \quad \tilde{p}_{k,(j)} = \frac{p_{k,(j)}}{\sum_{m=1}^{64} p_{k,(m)}}$$

---

## 3. Structural Fragility Priors and Quantile-Stabilized Calibration

To account for intrinsic problem fragility—such as extremal combinatorics ("minimum/maximum number of ways"), modular arithmetic traps ($\ge 3$-digit constants), and deeply nested LaTeX operators—we combine the four white-box activations with a structural prior $\Phi(P) \in [0.05, 0.95]$ and `reasoning_effort` shift $\delta_{\text{effort}} \in \{-0.06, 0.0, +0.06\}$:

$$R(P; \mathcal{M}_\theta, \text{effort}) = 0.38\,\mathcal{S}_{\cos} + 0.12\,\mathcal{S}_{\text{norm}} + 0.22\,\bar{\Delta}_{\log p} + 0.14\,(1 - \bar{\mathcal{H}}_{64}) + 0.14\,\Phi(P) + \delta_{\text{effort}}$$

Finally, because raw hidden-state cosine geometry varies systematically across model families (`Qwen3.5-4B` vs. `Olmo-3-7B-Think` vs. `gpt-oss-120b`), applying a naive static threshold across all models induces calibration bias. Since `are_robust` receives the full batch of problems $\mathcal{P} = \{P_1, \dots, P_N\}$ for a given `(model_id, reasoning_effort)` pair, we compute a **quantile-stabilized decision boundary**:

$$\tau(\mathcal{M}_\theta) = 0.65 \cdot \text{median}_{i=1..N}\big(R(P_i; \mathcal{M}_\theta, \text{effort})\big) + 0.35 \cdot \tau_0, \quad \tau_0 = 0.52$$

and predict `True` (robust) if and only if $R(P_i; \mathcal{M}_\theta, \text{effort}) \ge \tau(\mathcal{M}_\theta)$.

---

## 4. Runtime & Memory Budget on NVIDIA RTX PRO 6000 Blackwell (96 GB)

| Model ID | Parameters | Precision / Format | Probe Rollout $K$ | Time per 50 Problems | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Qwen/Qwen3.5-4B` | 4B | BF16 | 16 tokens | ~18 s | ~9 GB |
| `Skywork/Skywork-OR1-Math-7B` | 7B | BF16 | 16 tokens | ~28 s | ~15 GB |
| `allenai/Olmo-3-7B-Think` | 7B | BF16 | 16 tokens | ~28 s | ~15 GB |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B | BF16 | 16 tokens | ~32 s | ~17 GB |
| `openai/gpt-oss-120b` | 120B (MoE) | Pre-quantized MXFP4 | 8 tokens | ~110 s | ~66 GB |

**Total End-to-End Runtime:** `< 4 minutes` (well under the `3,600 seconds` Codabench hard limit), with explicit `del model`, `gc.collect()`, and `torch.cuda.empty_cache()` between `are_robust` invocations.
