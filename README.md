# Could Language Models Learn Impossible Indonesian Language?

**Author:** Bastian Farandy  
**Programme:** MSc in Artificial Intelligence, School of Informatics, University of Edinburgh (2025)  
**Supervisors:** Prof. Sharon Goldwater & Coleman Haley  
**Based on:** Julie Kallini et al. (*Mission: Impossible Language Models*, ACL 2024)

---

## 📌 Executive Summary & Motivation

In classical linguistic theory, Noam Chomsky posited that human language acquisition is governed by universal constraints (Universal Grammar)—humans can only naturally acquire structurally "possible" natural languages, while artificially constructed "impossible" languages violating hierarchical rules cannot be natively acquired. 

Recent debates questioned whether modern Large Language Models (LLMs) learn natural and unnatural/impossible languages with equal ease. Kallini et al. (2024) introduced the **Impossible Language Framework** using English, demonstrating that GPT-2 struggles significantly more to learn perturbed, structurally impossible languages than natural English.

However, English is typologically atypical and resource-dominant. This MSc dissertation investigates:
> **"Could Language Models Learn Impossible Indonesian Language?"**

Indonesian provides a compelling typological counterpoint:
1. **Morphological profile:** Simpler inflectional morphology (no verb tense inflections, pluralization via reduplication like *buku* $\rightarrow$ *buku-buku*) combined with rich derivational affixation (*meng-*, *di-*, *-kan*, *-i*, *ber-*).
2. **Word order dependencies:** Relies heavily on canonical word order (SVO) to encode syntactic roles and grammatical relations.

By translating the developmentally plausible **BabyLM** benchmark into Indonesian, applying systematic perturbations, pretraining GPT-2 architectures from scratch, and evaluating on Indonesian minimal pairs (**BHASA**), this project probes how language-specific typological properties alter the boundaries of neural language learnability.

---

## 🔬 Experimental Pipeline & Methodology

```
┌─────────────────────────┐
│   BabyLM English Data   │ (10M / 100M tokens: CHILDES, BNC, Wikipedia, Subtitles, etc.)
└────────────┬────────────┘
             │ Translation & Quality Filtering (NLLB-200 distilled 600M & Helsinki OPUS-MT)
             ▼
┌─────────────────────────┐
│  Indonesian BabyLM Data │ (Evaluated with COMET metric: Unbabel/wmt22-cometkiwi-da)
└────────────┬────────────┘
             │ Custom Tokenizer Training (Byte-level BPE, fertility analysis)
             ▼
┌─────────────────────────┐
│ Perturbation Generation │ (Deterministic Shuffles, Non-deterministic Shuffles, 
└────────────┬────────────┘  Local Windows k=3/5/10, Even-Odd, Partial/Full Reversals)
             │
             ▼
┌─────────────────────────┐
│ GPT-2 Small Pretraining │ (Trained via Mistral framework from scratch across multiple seeds)
└────────────┬────────────┘
             │
             ├───────────────────────────────────────────┐
             ▼                                           ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│ Perplexity & NLL Eval   │                 │ BHASA Minimal Pairs     │
│ (In-domain test sets)   │                 │ (Syntax & Morphology)   │
└─────────────────────────┘                 └─────────────────────────┘
```

---

## 🌀 Perturbation Typology ("Impossible Languages")

Following and adapting Kallini et al. (2024) to the Indonesian language structure:

| Perturbation Type | Description & Linguistic Disruption |
| :--- | :--- |
| **`shuffle_control`** | Unperturbed natural Indonesian baseline (*control*). |
| **`shuffle_deterministic`** | Words in each sentence are reordered using a fixed pseudo-random permutation sequence across seeds (21, 57, 84). Preserves 1-to-1 position mapping. |
| **`shuffle_nondeterministic`** | Words in every sentence are completely randomized without a consistent permutation pattern, destroying all positional regularity. |
| **`shuffle_local (k=3, 5, 10)`** | Words are shuffled within localized windows of size $k$, preserving coarse global ordering while disrupting local dependencies. |
| **`shuffle_even_odd`** | Words at even indices are collected first, followed by words at odd indices, breaking continuous constituent structure. |
| **`reverse_control`** | Unreversed baseline tagged with sequence markers. |
| **`reverse_partial`** | Randomly selected constituent clauses or phrases are reversed. |
| **`reverse_full`** | The entire sentence is reversed token-by-token from end to start. |

---

## 📊 Key Findings

1. **Deterministic Perturbations Are Easier to Learn in Indonesian than English:**
   - Under deterministic transformations (deterministic shuffle, even-odd, partial/full reversals, local window shuffles), GPT-2 achieved relatively lower perplexities in Indonesian compared to the equivalent English setups.
   - Distinctive Indonesian morphological markers (e.g., derivational affixes, reduplication markers) retain lexical-grammatical information even when linear positions are permuted systematically.

2. **Non-Deterministic Shuffling Severely Harms Indonesian:**
   - Random non-deterministic shuffling drastically degraded learning in Indonesian, performing significantly worse than in English and failing to converge in several runs.
   - Because Indonesian relies critically on fixed word order rather than inflectional case markings to determine thematic roles (who does what to whom), complete word randomness destroys syntactic interpretability.

3. **Linguistic Probing via BHASA Minimal Pairs:**
   - Evaluated on **BHASA** grammatical minimal pairs (covering filler-gap dependencies, voice morphology, argument structure, negative polarity items [NPIs], and negation).
   - Natural Indonesian models achieved the strongest syntactic discernment; applying structural perturbations precipitated an immediate drop in grammatical preference accuracy.
   - Negative Log-Likelihood (NLL) remained relatively stable across perturbations, whereas token perplexity fluctuated drastically, demonstrating that minimal-pair preference specifically isolates grammatical competence.

4. **Typological Conclusion:**
   - The learnability boundary between "possible" and "impossible" languages is **not universal across languages**—it shifts depending on typological properties (inflectional vs. derivational morphology, rigid vs. flexible word order).

---

## 📂 Repository Structure

```
.
├── code/
│   ├── translate.py               # Dataset translation pipeline (BabyLM EN -> ID)
│   ├── translation_nllb.py        # NLLB-200 distilled 600M batch translation
│   ├── translation_opus_mt.py     # OPUS-MT Helsinki-NLP translation
│   ├── comet_eval.py              # COMET metric evaluation for MT quality
│   ├── tag.py / original_tag.py   # Part-of-Speech and syntactic tagging for sentences
│   ├── perturb.py                 # Core perturbation engine (shuffling, reversals, hops)
│   ├── train_tokenizer.py         # Trains custom tokenizer from translated BabyLM 100M
│   ├── tokenizer_analysis.py      # Evaluates tokenizer fertility and compression ratios
│   ├── evaluate_bhasa.py          # Evaluates models on BHASA Indonesian minimal pairs
│   ├── evaluate_blimp.py          # English BLiMP evaluation script
│   ├── gpt_test.py                # Perplexity & NLL scoring across checkpoints
│   ├── helper.py / utils.py       # Shared constants, perturbation functions, and IO utilities
│   └── training/                  # YAML generation scripts for the Mistral GPT training pipeline
│
├── dataset/                       # Cleaned, translated, and perturbed dataset splits
├── results/                       # Evaluation outputs, CSV tables, and visualization plots
│   ├── syntax_evaluation/         # BHASA minimal pair syntax results
│   ├── blimp_evaluation/          # BLiMP results
│   └── *.png / *.csv              # Perplexity curves and perturbation comparisons
├── models/                        # Trained custom tokenizers and architecture configurations
├── submission/                    # Final MSc Dissertation artifacts
│   ├── B256818.pdf                # Full MSc Dissertation thesis document
│   ├── README.md                  # Quick submission guide
│   └── BHASA/                     # BHASA benchmark evaluation files
└── *.sh                           # Shell scripts automating end-to-end training and evaluation
```

---

## 🚀 Reproduction & Usage Guide

### 1. Translation & Quality Assessment
Translate BabyLM from English to Indonesian and calculate COMET scores:
```bash
bash translation_nllb.sh
# or: python code/translation_opus_mt.py
python code/comet_eval.py
```

### 2. POS Tagging & Perturbation Generation
Generate the impossible language variants:
```bash
bash data_prep_tag.sh
python code/perturb.py --seed 21
```

### 3. Custom Tokenizer Training
Train the Indonesian tokenizer to match the translated BabyLM vocabulary:
```bash
python code/train_tokenizer.py
python code/tokenizer_analysis.py
```

### 4. GPT-2 Pretraining
Generate model configurations and pretrain GPT-2 via Mistral:
```bash
cd code/training
python generate_yaml.py
cd ../..
bash gpt_train.sh
```

### 5. Evaluation (Perplexity & BHASA Minimal Pairs)
Evaluate checkpoint perplexity and syntactic capability:
```bash
# Perplexity and NLL
bash run_all_evaluations.sh

# BHASA Indonesian minimal pairs evaluation
python code/evaluate_bhasa.py --model_path <path_to_checkpoint>
```

---

## 📚 References & Acknowledgments

- **Julie Kallini, Isabel Papadimitriou, Richard Futrell, Kyle Mahowald, Christopher Potts (ACL 2024):** *Mission: Impossible Language Models*.
- **Alex Warstadt et al. (CoNLL 2023):** *Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora*.
- **Farhan Samir et al. (ACL 2023):** *BHASA: A Holistic Indonesian and Sundanese NLP Benchmark*.
- **Supervisor Guidance:** Prof. Sharon Goldwater & Coleman Haley, School of Informatics, University of Edinburgh.
- **Sponsorship:** Supported by the Indonesia Endowment Fund for Education (LPDP).
