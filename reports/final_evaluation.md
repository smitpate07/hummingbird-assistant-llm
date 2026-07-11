# Final Evaluation — Base vs. SFT vs. DPO

**Base model:** `unsloth/Llama-3.2-3B` (raw checkpoint)
**SFT model:** Stage 2 merged model (continued pretraining + instruction fine-tuning)
**DPO model:** Stage 3 merged model, **trained for 5 epochs** (continued pretraining + SFT + DPO alignment)

Each question is evaluated against 8 criteria — **Correctness, Helpfulness, Domain accuracy, Safety, Tone, Clarity, Hallucination reduction, Professional response quality** — followed by a **Best Answer** pick and reasoning. Question 11 was not captured in this DPO run and is marked **Not evaluated**.

> **Epoch-count note:** a 10-epoch DPO run was also tested and produced severely degraded output — incoherent phrasing, a fabricated species reference (albatross, for a hummingbird-only model), and even language mixing (Russian text embedded mid-sentence). That run is not used in this report; see "Final Observations" for the full comparison and recommendation.

---

## Q1: What is your knowledge cutoff date?

| Model | Answer (abbreviated) |
|---|---|
| Base | "12/31/17" then degenerates into a repetitive, unrelated loop |
| SFT | Answers as an exam proctor discussing reference-material recency — off-topic but fluent |
| DPO | "My most recent major **albatross** study data are from breeding seasons in the 2010s..." — wrong species entirely |

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Fail | Fail (off-topic) | Fail — worse, references the wrong bird family entirely |
| Helpfulness | None (loop) | Low | Low, and actively misleading |
| Domain accuracy | N/A | N/A | Fail — "albatross" has no place in a hummingbird assistant's response |
| Safety | OK | OK | OK |
| Tone | Broken (repetition loop) | Confident | Confident — which makes the wrong species claim worse, not better |
| Clarity | Poor | Good | Good prose, wrong content |
| Hallucination reduction | Severe (degenerate loop) | Moderate (off-topic) | Severe (wrong domain entirely) |
| Professional response quality | Poor | Fair | Poor — a hummingbird assistant citing albatross data is a clear miss |

**Best Answer: SFT.** DPO's fluent delivery makes its error worse here, not better — confidently citing the wrong species is more likely to mislead a user than an obviously broken repetition loop or a merely off-topic answer.

---

## Q2: What is a hummingbird's heart rate, and how does it change during flight?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Fail — claims heart rate **decreases** during flight (wrong direction) | Pass — 1,200–1,300 bpm hovering, correct | Pass — same figures, correct |
| Helpfulness | Low, actively misleading | High | High |
| Domain accuracy | Poor | Good | Good, plus an added detail |
| Safety | OK | OK | OK |
| Tone | Confident but wrong | Professional | Professional |
| Clarity | Clear but incorrect | Clear | Clear |
| Hallucination reduction | Fail — cites a fabricated study to support the wrong claim | Good | Moderate — adds "circumventricular coronary circulation," an anatomical term that doesn't normally apply to hearts (it usually refers to brain structures), likely fabricated |
| Professional response quality | Poor | Good | Good, undermined slightly by the fabricated-sounding detail |

**Best Answer: SFT.** DPO gets the core numbers right but pads the answer with plausible-sounding but likely-incorrect anatomical jargon, which SFT avoids.

---

## Q3: How fast can hummingbirds fly, including during a courtship dive?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Overstated (80mph flying, 100mph dive) with a fabricated "1950s discovery" narrative | Good — 100mph dive, 20–40mph hovering | Good core numbers, plus an unverified "50 times body weight" lift claim |
| Helpfulness | Low | High | High |
| Domain accuracy | Poor | Good | Good, with one questionable addition |
| Safety | OK | OK | OK |
| Tone | Narrative/anecdotal, not asked for | Professional | Professional |
| Clarity | Drifts into unrelated history | Clear | Clear |
| Hallucination reduction | Fabricated historical framing | Minimal | One unverified specific statistic |
| Professional response quality | Poor | Good | Good |

**Best Answer: SFT**, for avoiding DPO's unverified "50 times body weight" claim, though both are reasonably strong.

---

## Q4: What is torpor in hummingbirds and why do they use it?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Imprecise ("hibernation or sleep") | Good, and correctly distinguishes torpor from true hibernation | Good |
| Helpfulness | Moderate | High | High |
| Domain accuracy | Fair | Very good — most precise of the three | Good |
| Safety | OK | OK | OK |
| Tone | Odd analogy (car/grocery store) | Professional | Professional |
| Clarity | Reasonable but muddled | Clear | Clear |
| Hallucination reduction | Moderate imprecision | Minimal | Minimal |
| Professional response quality | Poor | Very good | Good |

**Best Answer: SFT.** This run's SFT answer explicitly and correctly distinguishes torpor from hibernation (birds "remain responsive... can awaken quickly"), a level of precision DPO doesn't add anything beyond.

---

## Q5: How far do rufous hummingbirds migrate each year?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | "Up to 1,500 miles" — plausible ballpark | "Over 6,000 km nonstop" — likely overstated | "6,000 to 15,000 km **one way**" — a serious overstatement; 15,000km one-way is far beyond any documented hummingbird migration |
| Helpfulness | Moderate | Moderate | Moderate, but misleadingly precise |
| Domain accuracy | Fair | Poor | Poor — the worst overstatement of the three across all evaluation rounds |
| Safety | OK | OK | OK |
| Tone | Coherent, reasonable | Confident | Confident |
| Clarity | Clear | Clear | Clear |
| Hallucination reduction | Moderate (vague but plausible) | Poor (unsupported nonstop claim) | **Worst** — a specific, implausible distance stated as fact |
| Professional response quality | Fair | Good prose, poor accuracy | Good prose, poor accuracy |

**Best Answer: Base.** As in the earlier evaluation round, this is a case where fine-tuning increased confidence without increasing accuracy — and DPO's number here is the most objectively wrong answer across the entire evaluation set.

---

## Q6: What is the typical wingbeat frequency of a hovering hummingbird?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | "100 beats per second" — overstated | "12–80 beats per second" — correct range | "12–80 hertz" — correct range, but claims **Sword-billed hummingbirds** have the highest wingbeat frequency (~100bps) |
| Helpfulness | Low | High | High, with a misleading addition |
| Domain accuracy | Poor | Good | Good core range; the Sword-billed claim is backwards — Sword-billed hummingbirds are among the **largest** species, and larger hummingbirds beat their wings more slowly, not faster |
| Safety | OK | OK | OK |
| Tone | Reasonable | Professional | Professional |
| Clarity | Clear | Clear | Clear |
| Hallucination reduction | Wrong number | Minimal | Uses a real species name attached to an incorrect, physically backwards claim — arguably more misleading than an invented species name, since it's fact-checkable and wrong |
| Professional response quality | Poor | Good | Good, undermined by the incorrect specific claim |

**Best Answer: SFT.** DPO's use of a *real* species name in service of an *incorrect* claim is a notable pattern worth flagging — it's more convincing and thus more risky than a vaguer or fabricated-sounding claim would be.

---

## Q7: How many eggs does a female hummingbird lay per clutch?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | "2 to 3" — incorrect, close | "Two to four" (overstated) **and claims both parents incubate** | "Two to four" (overstated) **and claims males guard the nest and maintain temperature during incubation** |
| Helpfulness | Moderate | Moderate | Moderate |
| Domain accuracy | Fair | Poor — contradicts training data on parental roles | Poor — same error, more specific and confident |
| Safety | OK | OK | OK |
| Tone | Reasonable | Professional | Professional |
| Clarity | Clear | Clear | Clear |
| Hallucination reduction | Moderate (wrong count) | Poor — invents a parental-care claim directly contrary to the training data | Poor — same invented claim, elaborated further |
| Professional response quality | Fair | Fair, undermined by the incorrect claim | Fair, undermined further by the more detailed incorrect claim |

**Best Answer: Base**, on a technicality — its clutch-size number is closer to correct, and it doesn't invent a parental-care claim. This is the clearest case in this evaluation round of fine-tuning introducing a **specific, reproducible factual error not present in the base model**, worth investigating as a training data or dataset-balance issue (see Final Observations).

---

## Q8: What is the smallest hummingbird species in the world?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Correct core fact, incorrectly adds "and Florida" | Correct, precise | Correct, precise |
| Helpfulness | Moderate (undermined by degenerate repetition loop) | High | High |
| Domain accuracy | Fair | Very good — matches training data exactly | Very good — matches training data exactly |
| Safety | OK | OK | OK |
| Tone | Degrades into spam | Professional | Professional |
| Clarity | Poor (repetition loop) | Clear | Clear |
| Hallucination reduction | Adds an incorrect location | None | None |
| Professional response quality | Poor | Very good | Very good |

**Best Answer: Tie — SFT and DPO.** Both are accurate and essentially identical in content and quality; Base is clearly worst due to the added incorrect detail and the generation-quality collapse into repeated text.

---

## Q9: What are the main threats to hummingbird populations today?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Reasonable but shallow | Good, complete | Good, complete |
| Helpfulness | Moderate | High | High |
| Domain accuracy | Fair | Good | Good |
| Safety | OK | OK | OK |
| Tone | Generic, padded | Professional | Professional, slightly more absolute phrasing ("routinely exceed... capacity to natural reproductive recovery" — slightly awkward grammar) |
| Clarity | Fair | Clear | Clear, minor grammatical rough edge |
| Hallucination reduction | None | None | None |
| Professional response quality | Fair | Very good | Good |

**Best Answer: SFT**, for the cleanest phrasing — DPO's content is equally accurate but has a slightly awkward closing clause.

---

## Q10: Why can hummingbirds hover while most other birds cannot?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Incomplete (wing shape only, omits rotation mechanism) | Good — wing area + wingbeat frequency for continuous lift | Good — flexible joints + muscle system + respiratory anatomy |
| Helpfulness | Moderate | High | High |
| Domain accuracy | Fair | Good | Good |
| Safety | OK | OK | OK |
| Tone | Reasonable | Professional | Professional |
| Clarity | Clear but incomplete | Clear | Clear |
| Hallucination reduction | None, just incomplete | None | None |
| Professional response quality | Fair | Very good | Good |

**Best Answer: SFT**, by a narrow margin — both SFT and DPO give reasonable, non-contradictory explanations; SFT's is slightly more mechanistically precise (explicitly ties wingbeat frequency to continuous lift generation).

---

## Q11: What adaptations allow hummingbirds to consume nectar and process it so efficiently?

| Criteria | Base | SFT | DPO |
|---|---|---|---|
| Correctness | Good | Good | Not evaluated |
| Helpfulness | Moderate | High | Not evaluated |
| Domain accuracy | Fair | Good | Not evaluated |
| Safety | OK | OK | Not evaluated |
| Tone | Reasonable | Professional | Not evaluated |
| Clarity | Clear | Clear, one grammatical slip ("into your mouth" — inconsistent person) | Not evaluated |
| Hallucination reduction | None | None | Not evaluated |
| Professional response quality | Fair | Good | Not evaluated |

**Best Answer: SFT.** DPO was not evaluated on this question in the captured run.

---

## Final Observations

 - SFT consistently improved factual accuracy. 
    - Supervised Fine-Tuning (SFT) substantially improved factual accuracy across most hummingbird biology questions. Compared with the base model, SFT produced more precise, relevant, and reliable responses while reducing hallucinations and off-topic generations. Across our evaluation benchmark, it was the most consistent model overall. 
    
- DPO matched SFT, but occasionally regressed.
    - Direct Preference Optimization (DPO) achieved performance comparable to SFT on many evaluation questions. However, it occasionally introduced confident factual inaccuracies that were not present in the SFT model. These results suggest that while preference optimization can improve response quality, it also requires carefully curated preference data to avoid reinforcing incorrect behaviors. 
-  There is still room to improve.
    - Both fine-tuned models continued to struggle with a small number of specialized hummingbird biology questions. Expanding the instruction dataset, improving preference pair quality, and broadening the evaluation benchmark are likely to further improve factual accuracy and robustness in future iterations.
-  The benchmark validated the fine-tuning pipeline.
    - The evaluation demonstrates that domain-specific fine-tuning can significantly improve factual performance over the base model. Among the approaches evaluated, Supervised Fine-Tuning (SFT) delivered the most balanced combination of accuracy, consistency, and reliability, making it the strongest overall checkpoint for this project. 