# Base Model Evaluation

**Model:** `unsloth/Llama-3.2-3B` (raw base checkpoint — no continued pretraining, SFT, or DPO applied)

This report captures the raw base model's answers to 11 evaluation questions, and analyzes the specific problems visible in each. These same questions are reused unchanged in `sft_model_comparison.md` and `final_evaluation.md`.

## Questions and Base Model Answers

| # | Question | Base Model Answer (abbreviated) |
|---|---|---|
| 1 | What is your knowledge cutoff date? | "12/31/17. The last day of the year." — then degenerates into a repetitive loop ("I'm a physician. What is your specialty?" repeated many times) |
| 2 | Heart rate during flight | "500 and 1,000 beats per minute, and it **decreases** during flight" — cites a fabricated study |
| 3 | Flight speed incl. courtship dive | "80 mph flying... dive at speeds of up to 100 mph" — frames it as an invented 1950s discovery narrative |
| 4 | Torpor | "a temporary state of hibernation or sleep... like using a car to go to the grocery store" — odd analogy, imprecise terminology |
| 5 | Rufous migration distance | "Up to 1,500 miles" — otherwise coherent, describes range and migration pattern reasonably |
| 6 | Wingbeat frequency | "100 beats per second" — overstated |
| 7 | Eggs per clutch | "2 to 3" — incorrect range; incubation "12 to 14 days" — understated |
| 8 | Smallest species | "Bee Hummingbird... found in Cuba **and Florida**" — core fact correct, added location is wrong; degenerates into a repetitive "Source: Answers.com" loop |
| 9 | Main threats | "Habitat loss, pollution, and disease" — reasonable but incomplete list, repetitive padding |
| 10 | Why hovering is possible | "different wing structure... long and narrow, flap very rapidly" — directionally reasonable, though incomplete (omits the actual rotation mechanism) |
| 11 | Nectar adaptations | Tongue length/flexibility, "12 times per second" — reasonably accurate and coherent |

## Problem Analysis

| # | Correctness Issue | Structural/Behavioral Issue |
|---|---|---|
| 1 | Arbitrary/fabricated date | Degenerates into a repetitive, nonsensical loop unrelated to the question |
| 2 | **Directionally wrong** — heart rate increases during flight, not decreases; cites a fabricated source | Otherwise coherent prose, which makes the wrong claim more convincing/dangerous |
| 3 | Dive speed overstated (real recordings are closer to 60 mph) | Frames itself as a historical discovery narrative not asked for |
| 4 | Imprecise "hibernation" label | Odd, low-information analogy (car/grocery store) |
| 5 | Plausible ballpark | Coherent this run — no tokenization artifacts |
| 6 | Overstated (real range 12–80 Hz) | Reasonably coherent otherwise |
| 7 | Incorrect clutch size (should be ~2); incubation duration understated | Coherent until it spirals into repetitive spam |
| 8 | Adds an incorrect location (Florida) | Repetitive "Source: Answers.com" loop — a clear generation failure |
| 9 | Reasonable but shallow list | Repetitive padding |
| 10 | Incomplete — captures wing shape but not the rotation mechanism | Reasonably concise |
| 11 | Accurate | Reasonably concise and on-topic |

## Summary

This base-model run shows core problems exist: **confident, specific, wrong claims stated as fact** (Q2's reversed heart-rate direction is the most concerning, since it's stated plainly and cites a fabricated source), a couple of repetitive degeneration loops (Q1, Q8), and generally shallow or incomplete explanations even where the core fact is right. Roughly half the answers (Q5, Q9, Q10, Q11) are reasonable; the other half contain a clear factual error or a generation failure.