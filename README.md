# Hummingbird Assistant LLM

A domain-specialized LLM assistant for hummingbird biology, behavior, ecology, physiology, and conservation — fine-tuned from `unsloth/Llama-3.2-3B` using QLoRA across three stages (continued pretraining → instruction fine-tuning → DPO alignment), trained end-to-end on a single Lightning.ai T4 Studio (16GB VRAM).

---

## 1. Domain Selected

**Hummingbird biology and conservation.** This spans flight mechanics, metabolism, torpor, foraging and pollination ecology, migration, breeding biology, hovering physiology, species diversity, and conservation threats — with the pretraining corpus specifically grounded in population health, disease ecology, and genomics research (Ernest et al., *Annual Review of Animal Biosciences*, 2024, CC BY 4.0).

## 2. Business Problem

General-purpose LLMs give shallow, generic, or occasionally incorrect answers on narrow biological domains — confusing hummingbird torpor with illness, giving vague foraging advice that ignores flower-bill coevolution, or citing outdated or wrong conservation figures. This project builds a small, cheap-to-run, domain-specialized assistant for:

- **Hummingbird enthusiasts / birders** who want more depth than a field guide provides

## 3. Dataset Details

| File | Purpose | Size | Format |
|---|---|---|---|
| `data/non_instruction_data.txt` | Continued pretraining corpus | 52 paragraphs | Plain text |
| `data/instruction_dataset.jsonl` | Supervised fine-tuning (SFT) | 100 instruction/response pairs | JSONL |
| `data/preference_dataset.jsonl` | DPO alignment | 65 preference pairs | JSONL: `{prompt, chosen, rejected}` |

The pretraining corpus is original-authored text, synthesized (not copied) from a real, CC BY 4.0-licensed research paper on hummingbird population health, disease ecology, and genomics. The instruction and preference datasets are original-authored, covering the full domain topic list above.

## 4. Base Model Used

- **Model:** `unsloth/Llama-3.2-3B` (base, non-instruction-tuned, bnb-4bit prequantized)
- **Why:** Upgraded from an initial `unsloth/Llama-3.2-1B` plan once training moved to Lightning.ai's T4 (16GB), which had comfortable headroom for a larger base model at the same QLoRA hyperparameters. 

## 5. Non-Instruction Fine-Tuning Approach

Stage 1 trains on `data/non_instruction_data.txt` with a plain causal language-modeling objective (`SFTTrainer` with `packing=True`, no instruction template) — the model just learns to predict the next token over hummingbird-domain text, nudging its internal vocabulary and associations toward the domain before any instruction-following is introduced.

## 6. Instruction Fine-Tuning Approach

Stage 2 starts from Stage 1's merged model and trains on `data/instruction_dataset.jsonl`, with each `{instruction, response}` pair formatted as:
```
### Instruction:
{instruction}

### Response:
{response}
```
Trained with `SFTTrainer` and `packing=False` (to preserve each example's prompt/response boundary), at a lower learning rate than Stage 1.

## 7. DPO Alignment Approach

Stage 3 starts from Stage 2's merged model and trains on `data/preference_dataset.jsonl` using `DPOTrainer`, learning to prefer the `chosen` response over the `rejected` response for the same prompt. No separate reference model is loaded — since the model is a PEFT/LoRA adapter, `trl` uses the adapter-disabled base as the implicit reference, saving VRAM.

## 8. LoRA / QLoRA Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Quantization | 4-bit (`load_in_4bit=True`) |
| Max sequence length | 512 |
| Batch size (per device) | 2 |
| Gradient accumulation steps | 4 |
| Learning rate — Stage 1 | 2e-4 |
| Learning rate — Stage 2 | 1e-4 |
| Learning rate — Stage 3 (DPO) | 5e-5 |
| DPO beta | 0.05 |
| Epochs — Stage 1 | 3 |
| Epochs — Stage 2 | 20 |
| Epochs — Stage 3 (DPO) | 3 (see Final Observations — higher epoch counts were tested and degraded output quality) |

See `reports/fine_tuning_explanation.md` for the reasoning behind each value.

## 9. Training Screenshots or Logs



## 10. Before vs. After Output Comparison

Real outputs from the evaluation reports (`reports/base_model_evaluation.md`, `reports/sft_model_comparison.md`, `reports/final_evaluation.md`).

**Q: What is a hummingbird's heart rate, and how does it change during flight?**

| Stage | Answer |
|---|---|
| Base | *"...between 500 and 1,000 beats per minute, and it **decreases** during flight..."* — directionally wrong, cites a fabricated study |
| SFT | *"...can exceed 1,200 to 1,300 beats per minute during hovering, some of the highest heart rates of any vertebrate..."* — correct, concise |
| DPO | Same correct figures, plus one likely-fabricated anatomical detail — see full report |

**Q: How many eggs does a female hummingbird lay per clutch?**

| Stage | Answer |
|---|---|
| Base | *"2 to 3."* — close but incorrect (true answer is almost universally exactly 2) |
| SFT | *"...a clutch of two to four... both parents participate in incubation..."* — clutch size still overstated, and **incorrectly claims both parents incubate** (contradicts training data — males don't participate in nest care) |
| DPO | Same overstated clutch size, elaborates the same incorrect parental-care claim further |

This second example is intentionally included as a **known, unresolved limitation** — see Final Observations.

## 11. Final Observations

- **SFT delivered the largest, most reliable quality jump** — coherence, directness, and domain accuracy all improved sharply and consistently.
- **DPO's improvement over SFT was inconsistent, and highly epoch-sensitive.** A 10-epoch DPO run produced incoherent, collapsed output (fabricated terminology, embedded foreign-language text, answers referencing the wrong bird family entirely). A 5-epoch run was coherent but still showed a smaller version of the same failure pattern on some questions. The safe range for DPO on this size of preference dataset (65 pairs) appears to be **at or below ~3 epochs**.
- **A specific, reproducible regression appears at Q7 (clutch size / parental care) in both SFT and DPO**, independent of epoch count: both invent parental incubation roles that directly contradict the instruction dataset. This points to a dataset-balance or repetition issue rather than a training-duration issue, and is flagged as open follow-up work rather than papered over.
- Full per-question, per-criterion analysis is in `reports/final_evaluation.md`.

## 12. Challenges Faced

- **Environment migration mid-project:** training moved from Google Colab to Lightning.ai partway through, requiring removal of all `google.colab`-specific code (Drive mounting, file upload widgets) and replacing it with Lightning.ai's Teamspace persistent-storage pattern.
- **HF Hub README validation errors:** Unsloth's `save_pretrained_merged`/`save_pretrained` auto-generates a `README.md` with a `base_model` metadata field — when a model is loaded from a local filesystem path (as later stages do, loading the previous stage's local checkpoint), that local path leaks into the metadata and gets rejected by the Hugging Face Hub's YAML validator on upload. Fixed with a `_fix_readme_base_model()` helper that rewrites the field to a valid Hub ID before uploading.
- **DPO epoch sensitivity:** unlike SFT, DPO on a small preference dataset degrades sharply past a certain epoch count, discovered empirically by comparing 3, 5, and 10-epoch runs rather than assumed upfront.
- **Persistent, training-data-contradicting hallucinations:** some specific facts (clutch size, parental incubation roles) failed to transfer correctly even after both SFT and DPO, despite being explicitly stated in the training data.

## 13. Future Improvements

- Investigate the Q7-style regression directly — try increasing repetition of the specific fact in the training data, or adding more preference pairs that directly contrast it against the model's likely prior (many bird species do share incubation duties).
- Re-run DPO at 1–3 epochs with checkpoint-by-checkpoint evaluation to find the actual quality peak, rather than relying on a small number of discrete epoch-count tests.
- Expand `non_instruction_data.txt` to cover the full topic breadth (flight mechanics, migration, foraging, etc.), not just disease/genomics.
- Add automated factual-consistency scoring (e.g., a larger LLM as judge) to replace/supplement manual side-by-side comparison.
- Add retrieval-augmented generation (RAG) over a curated hummingbird literature corpus as a complementary approach to pure fine-tuning, particularly for citation-level accuracy.

## 14. Repository Structure

```
hummingbird-assistant-llm/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── base_model_evaluation.ipynb
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── README.md
└── requirements.txt
```

## 15. Storage Layout

All three stages save to a Lightning.ai Teamspace path (persistent local storage) and a **single shared Hugging Face Hub repo**, organized by subfolder rather than one repo per stage:

```
{HF_REPO_FULL}/
├── stage1-adapter/
├── stage1-merged/
├── stage2-adapter/
├── stage2-merged/
├── stage3-adapter/
└── stage3-merged/
```

## 16. How to Run

1. Open `notebooks/base_model_evaluation.ipynb` on Lightning.ai (T4 Studio) to establish the baseline.
2. Run `notebooks/non_instruction_finetuning.ipynb` (Stage 1).
3. Run `notebooks/instruction_finetuning.ipynb` (Stage 2), which loads Stage 1's merged model automatically.
4. Run `notebooks/dpo_alignment.ipynb` (Stage 3), which loads Stage 2's merged model automatically.
5. Run inference via `src/inference.py`, pointing at the final Stage 3 merged model (Teamspace path or `{HF_REPO_FULL}/stage3-merged` on the Hub).