# Fine-Tuning Explained

A plain-language guide to the techniques and hyperparameters used to build the Hummingbird Assistant LLM.

## The Techniques

### LoRA (Low-Rank Adaptation)
Instead of updating all of a model's billions of parameters during training, LoRA freezes the original model and inserts small, trainable "adapter" layers alongside it. Only those small layers get updated. This makes training dramatically faster and cheaper, while producing a tiny adapter file (megabytes, not gigabytes) that can be attached to the base model at inference time.

### QLoRA (Quantized LoRA)
QLoRA adds one more trick: it shrinks the base model itself down to 4-bit precision (instead of the usual 16 or 32-bit) before attaching LoRA adapters. This cuts memory use dramatically, which is what makes it possible to fine-tune a multi-billion-parameter model on a single consumer-grade GPU like a T4, rather than needing a data center.

### Non-Instruction Fine-Tuning (Continued Pretraining)
This is the first training stage. The model is shown plain domain text — no questions, no answers, just paragraphs — and learns to predict the next word, the same way it was originally pretrained. The goal isn't to teach it to follow instructions yet; it's to nudge its internal knowledge and vocabulary toward the hummingbird domain before anything else happens.

### SFT (Supervised Fine-Tuning)
This is the second stage, where the model learns to actually behave like an assistant. It's shown example question/answer pairs and learns to produce a direct, relevant answer instead of just continuing text. This is the stage most responsible for turning a raw language model into something that feels like a chatbot.

### DPO (Direct Preference Optimization)
This is the third and final stage. Instead of showing the model single correct answers, it's shown pairs — a *better* answer and a *worse* answer to the same question — and trained to prefer the better one. This is a lighter-weight alternative to RLHF (Reinforcement Learning from Human Feedback): it gets a similar "align the model with what we actually want" effect without needing a separate reward model or reinforcement learning loop.

*ORPO is a related, newer technique that combines the SFT and preference-alignment steps into a single training pass. This project uses the more traditional separate SFT-then-DPO pipeline instead, since it makes it easier to evaluate each stage's contribution independently.*

## Hyperparameters Used, and Why

| Hyperparameter | Value | Why |
|---|---|---|
| LoRA rank (`r`) | 16 | Controls the size of the adapter layers. 16 is a common, well-tested middle ground — big enough to learn meaningfully, small enough to stay fast and memory-cheap. |
| LoRA alpha | 32 | Scales how much influence the adapter has relative to the frozen base model. A 2x-rank ratio (32 = 2 × 16) is a standard, stable default. |
| LoRA dropout | 0.05 | A small amount of regularization to reduce overfitting, without slowing down training much. |
| Learning rate — Stage 1 | 2e-4 | Continued pretraining on plain text can tolerate a relatively higher learning rate since there's no risk of breaking instruction-following behavior that doesn't exist yet. |
| Learning rate — Stage 2 | 1e-4 | Lower than Stage 1, since the model is now learning a more specific behavior (instruction-following) that's easier to overshoot or destabilize with too-large updates. |
| Learning rate — Stage 3 | 5e-5 | Lowest of the three. DPO fine-tunes *preferences* on top of already-learned behavior, and is the most sensitive to being pushed too hard, too fast. |
| Batch size | 2 (per device) | Kept small to fit comfortably in GPU memory alongside a multi-billion-parameter model in 4-bit precision. |
| Gradient accumulation steps | 4 | Simulates a larger effective batch size (2 × 4 = 8) without needing the memory a true batch of 8 would require. |
| Max sequence length | 512 | Long enough for a solid paragraph or a detailed Q&A pair, short enough to keep memory use and training speed manageable. |
| DPO beta | 0.05 | Controls how far the model is allowed to drift from its Stage 2 starting point while learning preferences. Lower values allow a stronger preference signal; this was chosen to make the preference training clearly noticeable rather than overly conservative. |
| Epochs — Stage 1 | 3 | A small number of passes is usually enough for continued pretraining to shift vocabulary and tone without overfitting to a relatively small text corpus. |
| Epochs — Stage 2 | 20 | Instruction datasets are often small (as this one is), so more passes are needed for the model to reliably learn the instruction-following pattern. |
| Epochs — Stage 3 | 5 | DPO trains on preference *pairs* rather than plain examples, and tends to be more sensitive to overtraining than SFT on the same amount of data, so a lower epoch count is used here by design. |