# Pattern-based Logical Fallacy Classification using Decoder-only Large Language Models

This anonymized repository contains the supplementary material associated with the paper submitted to Argumentation Mining 2026 workshop.

## Contents

- [`prompts/`](./prompts/): prompt templates used in the experiments
  - [`prompts.yaml`](./prompts/prompts.yaml): zero-shot classification prompts 
  - [`prompts-one-shot.yaml`](./prompts/prompts-one-shot.yaml): one-shot classification prompts
  - [`prompts_generation.yaml`](./prompts/prompts_generation.yaml): pattern extraction prompts
 
  
- [`features/`](./features/): features used in the experiments, organized per dataset:
  - The following four features are originally derived for the Logic dataset and restricted to the relevant fallacy classes of each dataset for the experiments reported in Section 5:
    - `definitions.json`: fallacy definitions sourced from [Lei and Huang, 2024](https://aclanthology.org/2024.emnlp-main.730.pdf)  (**DEF**)
    - `logical_forms.json`: logical forms sourced from [LogicallyFallacious](https://www.logicallyfallacious.com) (**LOGICAL FORMS**)
    - `generated_definitions.json`: LLM-derived fallacy definitions extracted from LOGIC (**NEW DEF**)
    - `logic_generated_patterns.json`: LLM-derived fallacy patterns derived from LOGIC (**PATTERNS**)
  - `dynamic_examples.json`: examples dynamically selected via SBERT, used in one-shot experiments (**DYNAMIC ONE-SHOT**, etc.)
  - `syntax_examples_roberta.json` and `syntax_examples_sbert.json`: examples selected by  SBERT and syntax-aumented ROBERTA-large for the syntax-based approach (**SYNTAX-BASED DYNAMIC ONE-SHOT**, LOGIC dataset only)
  - `elecdebate/elecdebate_patterns.json` and `reddit/reddit_patterns.json`: LLM-derived fallacy patterns extracted from dataset-specific arguments, used in Section 5 (**SAME DATASET PATTERNS**)
 
  
## Anonymization

This repository has been anonymized for double-blind review using [Anonymous GitHub](https://anonymous.4open.science/).
