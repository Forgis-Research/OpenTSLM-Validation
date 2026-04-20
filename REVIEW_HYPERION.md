# Overnight Review Prompt: Hyperion Paper

**Paper**: `Hyperion__Time_Series_Understanding_via_Discrete_Tokenization.pdf` (in this repo root)

Run this prompt with Claude Code to get a comprehensive review, explanation, and improvement plan.

---

## Prompt

Read the full paper at `Hyperion__Time_Series_Understanding_via_Discrete_Tokenization.pdf` (all pages). Then execute the following tasks sequentially, writing all output to `review_output.md` in this repo root.

### Task 1: Four Independent NeurIPS Reviews

Produce 4 independent, rigorous reviews of this paper using the neurips-reviewer agent persona. Each review should be written from a different reviewer archetype:

1. **Reviewer 1 (Time-Series Foundation Model Expert)** - Deep knowledge of Chronos, TimesFM, Moirai, Lag-Llama, TOTEM, Mantis. Focus on positioning relative to the TSFM landscape, missing baselines, whether VQ-VAE tokenization is truly novel vs. incremental over TOTEM+Chronos.

2. **Reviewer 2 (Multimodal LLM Researcher)** - Expertise in Flamingo, LLaVA, Chameleon, early-fusion vs cross-attention. Focus on whether the "cross-attention fails" claim is adequately supported, whether 14 OpenTSLM configs are enough, whether the sensitivity test is a valid diagnostic.

3. **Reviewer 3 (Applied ML / Industrial Deployment)** - Cares about real-world impact, latency, cost, deployment feasibility. Scrutinize the Forgis deployment section (Section 4.4) for missing details, the practical value of NL explanations, and whether 86-89% accuracy is actually useful in safety-critical domains.

4. **Reviewer 4 (Methodology & Statistics Hawk)** - Focuses on experimental rigor. Check: are error bars reported? How many seeds? Is n=200 sufficient? Are the sensitivity test and output divergence metrics well-defined? Is the scalar-binning vs VQ-VAE comparison confounded? Are the "TBD" entries in Table 2 acceptable?

For each review, follow the full NeurIPS review form: Summary, Strengths (3-6), Weaknesses (3-6 with fixes), Questions for Authors, Missing References, Minor Issues, and all Scores (Soundness, Significance, Novelty, Clarity, Reproducibility, Overall 1-10, Confidence 1-5).

### Task 2: Paper Explanation (for the authors)

Write a 1500-word explanation of the paper aimed at the authors themselves, covering:

1. **What the paper actually says vs. what it thinks it says** - Identify any gaps between the narrative framing and what the experiments actually demonstrate.
2. **The strongest argument in the paper** - What is the single most compelling thing here? (Likely the sensitivity test showing cross-attention ignores the signal.)
3. **The weakest link** - What is the single biggest vulnerability a reviewer will attack?
4. **Hidden assumptions** - What does the paper assume that it never states?
5. **The real contribution** - Strip away the framing. What did this paper actually add to the field?

### Task 3: Acceptance Probability Estimate

Based on the 4 reviews, estimate:
- **NeurIPS 2025 main conference**: probability of acceptance (%) with reasoning
- **NeurIPS 2025 Datasets & Benchmarks track**: probability (%) with reasoning
- **ICML 2025**: probability (%) with reasoning
- **ICLR 2026**: probability (%) with reasoning

Factor in:
- The paper has multiple "TBD" and "[WIP]" placeholders in Table 2 and elsewhere
- The Limitations section is empty (just "1.")
- The Forgis deployment section has "[FORGIS-FILL-IN]" placeholders throughout
- Table 1 has missing percentage values in the Share column
- Several appendix cross-references are broken ("??")

### Task 4: Prioritized Improvement Plan

Using the paper-writer agent persona, produce a concrete, prioritized action plan:

#### Tier 1: Must-Fix Before Any Submission (Rejection-Guaranteeing Issues)
- List every incomplete section, broken reference, TBD entry, and placeholder
- For each, specify exactly what needs to be done

#### Tier 2: High-Impact Improvements (Would Move Score +1 to +2 Points)
- Missing baselines to add (specific models, specific benchmarks)
- Statistical rigor gaps (error bars, seeds, significance tests)
- Evaluation scope expansions
- Figure 1 redesign (currently architecture diagram - should show the sensitivity test finding)

#### Tier 3: Polish (Would Move Score +0.5 Points)
- Writing improvements (specific paragraphs that are weak)
- Better framing of the "lobotomization" finding
- Strengthening the narrative arc
- Abstract tightening

#### Tier 4: Stretch Goals (Best Paper Territory)
- What would make this a best paper candidate?
- What additional experiments would be transformative?
- How to make the sensitivity test into a standalone contribution?
- Coined terminology improvements ("Hyperion" is fine but "sensitivity test" is generic)

### Task 5: Detailed Section-by-Section Feedback

Go through every section of the paper and provide specific, actionable feedback:

For each section, note:
- **Keep**: What works well and should be preserved
- **Cut**: What is filler or redundant
- **Add**: What is missing
- **Rewrite**: Specific sentences/paragraphs that need rework (quote the original, suggest the fix)

Sections to cover:
1. Title & Abstract
2. Introduction (Section 1)
3. Related Work (Section 2)
4. Method (Section 3) - Tokenization, Sensitivity Test, Training
5. Experiments (Section 4) - Setup, Main Results, TSQA, Forgis Deployment
6. Discussion (Section 5)
7. Limitations (Section 5.1)
8. Conclusion (Section 6)
9. Appendices (A, B, C)

### Task 6: Competitive Landscape Analysis

Search the web for the latest papers (2024-2025) in:
- Time series foundation models (especially any that use discrete tokenization)
- LLM-based time series reasoning
- Multimodal time series understanding
- VQ-VAE for time series

For each relevant paper found, assess:
- Does it threaten Hyperion's novelty claims?
- Should it be cited and compared against?
- Does it strengthen or weaken Hyperion's positioning?

Pay special attention to:
- Chat-TS (Quinlan et al., 2025) - closest competitor, how does Hyperion differentiate?
- Mantis (Feofanov et al., 2025) - ViT-based TSFM, relevant baseline?
- Any new work on the "LLMs ignore time series" problem
- Any work that has independently proposed sensitivity-test-like diagnostics

### Task 7: Rewrite Suggestions for Key Sections

Provide draft rewrites for:

1. **Abstract** - Tighter, more precise, lead with the sensitivity test finding
2. **Introduction opening paragraph** - The clinician/engineer vignettes are good but could be sharper
3. **Contributions list** - Reorder to lead with the strongest (sensitivity test)
4. **Limitations section** - Write a proper one (currently empty)
5. **A coined name for the sensitivity test** - "Sensitivity test" is too generic. Propose 3-5 memorable alternatives that could become the paper's lasting contribution (e.g., "Signal Fidelity Test", "Modality Grounding Diagnostic", etc.)

---

## Output Format

Write everything to `review_output.md` with clear headers and horizontal rules between sections. Use markdown formatting throughout. Total output should be 8,000-15,000 words.
