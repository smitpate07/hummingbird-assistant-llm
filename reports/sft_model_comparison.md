# SFT Model Comparison (Base vs. Stage 1 vs. Stage 2 Instruction Fine-Tuning)

**Base model:** `unsloth/Llama-3.2-3B` (raw checkpoint)
**Stage 1 model:** continued-pretrained merged model (raw-text domain adaptation, disease/genomics/population-health corpus)
**Stage 2 (SFT) model:** Stage 1 + instruction fine-tuning on `data/instruction_dataset.jsonl`

Same 11 questions as `base_model_evaluation.md`, run against all three checkpoints.

## Side-by-Side Answers

| # | Question | Base | Stage 1 (Non-Instruction) | Stage 2 (SFT) |
|---|---|---|---|---|
| 1 | Knowledge cutoff | Fabricated date, degenerates into a repetitive loop | Answers as if about disease-surveillance "documentation" cutoffs — off-topic but on-brand for the disease/genomics corpus | Answers as an exam proctor discussing reference-material recency — still off-topic, most fluent of the three |
| 2 | Heart rate | "500–1,000 bpm, **decreases** during flight" — wrong direction | Doesn't actually answer with numbers — discusses telemetry tag limitations instead | "1,200–1,300 bpm during hovering" — correct, concise |
| 3 | Flight speed | "80 mph flying, 100 mph dive" — overstated, fabricated discovery narrative | Cites real comparison species (Anna's hummingbird ~60mph) but drifts into unrelated bumblebee/course-reading content | "100 mph dive, 20–40 mph hovering" — plausible, well-structured |
| 4 | Torpor | "hibernation or sleep" — imprecise, odd analogy | Reasonably accurate general description of torpidity, but cuts off mid-sentence | Correct, and specifically **distinguishes torpor from true hibernation** (responsive, can wake quickly) — the most precise answer of the three |
| 5 | Migration distance | "Up to 1,500 miles" — plausible ballpark | "Up to 400 km" — likely understated, hedged with appropriate uncertainty language | "Over 6,000 km nonstop" — confident but likely overstated |
| 6 | Wingbeat frequency | "100 beats per second" — overstated | Cites "120–140 BPM" — **confuses wingbeat frequency with heart rate**, a notable category error | "12–80 beats per second" — matches literature, correct |
| 7 | Eggs per clutch | "2 to 3" — incorrect | Doesn't give a number — discusses difficulty of measuring brood size, cites "two to eight young" as a range | "Two to four" eggs (still overstated) — **and incorrectly claims "both parents participate in incubation"**, contradicting the training data |
| 8 | Smallest species | Correct core fact, incorrectly adds "and Florida" | Correct, adds real ecological detail (interactions with crows/vultures) | Correct and precise — matches training data exactly (1.6–2.6g, Cuba only) |
| 9 | Main threats | Reasonable but shallow list | Correct list, adds specific (though somewhat garbled) detail about introduced species | Correct, concise, matches training data closely |
| 10 | Why hovering is possible | Reasonable but incomplete (wing shape only) | Attributes it partly to bill shape/foraging needs — a plausible but non-standard framing | Correct — wing area and wingbeat frequency generating continuous lift |
| 11 | Nectar adaptations | Correct and coherent | Correct (tongue extension, high metabolic rate) | Mostly correct; one grammatical slip (switches to second-person "your mouth" mid-sentence) |

## Key Observations

- **Stage 2 (SFT) is the clearest, most consistently accurate of the three** — 8 of 11 answers are correct and well-formed, versus roughly half for Base and Stage 1.
- **Stage 1 shows its training data's fingerprint clearly** — its answers lean toward hedged, citation-heavy academic language (matching the disease/genomics corpus's style) even on questions the corpus doesn't directly cover (flight speed, wingbeat frequency), and it sometimes avoids giving a direct number where Base and SFT do.
- **Stage 1 introduced a notable category error at Q6**, confusing wingbeat frequency with heart rate (120–140 BPM) — worth checking whether this stems from the corpus's heavy emphasis on cardiovascular/heart-rate figures bleeding into an unrelated question.
- **A new, concerning finding at Q7**: SFT confidently states "both parents participate in incubation," which directly contradicts the instruction dataset's explicit statement that males do not participate in nest care. This is a real regression worth investigating — it's not a case of the model failing to learn a fact, but of it learning something specifically contrary to what the training data says.
- **Q1 (knowledge cutoff) remains unresolved across all three** — none of the checkpoints answers this as a question about itself in a clearly correct way.