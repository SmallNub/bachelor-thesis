# Multi-Step Semantic Reasoning in Generative Retrieval

This repository hosts the source code used by the conference paper and the bachelor thesis.

The original title of the bachelor thesis: "The Integration of Chain-of-Thought in Generative Information Retrieval".

The new title of the conference paper (ECIR2026): "Multi-Step Semantic Reasoning in Generative Retrieval".

## Abstract

Generative retrieval (GR) models encode a corpus within model parameters and generate relevant document identifiers directly for a given query. While this paradigm shows promise in retrieval tasks, existing GR models struggle with complex queries in numerical contexts, such as those involving semantic reasoning over financial reports, due to limited reasoning capabilities. This limitation leads to suboptimal retrieval accuracy and hinders practical applicability.
We propose ReasonGR, a framework designed to enhance multi-step semantic reasoning in numerical contexts within GR. ReasonGR employs a structured prompting strategy combining task-specific instructions with stepwise reasoning guidance to better address complex retrieval queries. Additionally, it integrates a reasoning-focused adaptation module to improve the learning of reasoning-related parameters.
Experiments on the FinQA dataset, which contains financial queries over complex documents, demonstrate that ReasonGR improves retrieval accuracy and consistency, indicating its potential for advancing GR models in reasoning-intensive retrieval scenarios.

For more information, see the published paper in the official proceedings of ECIR2026 (48th European Conference on Information Retrieval).

The conference paper: [https://link.springer.com/chapter/10.1007/978-3-032-21300-6_17](https://link.springer.com/chapter/10.1007/978-3-032-21300-6_17)

The preprint version: [https://arxiv.org/abs/2603.12368](https://arxiv.org/abs/2603.12368)

The bachelor thesis: [https://scripties.uba.uva.nl/search?id=record_56556](https://scripties.uba.uva.nl/search?id=record_56556)

If there are any questions, feel free to open an issue.

## Requirements

- CUDA Hardware that can handle TF32 and BF16 computations

## Guides

### Project Initialization

1. Create a python environment (venv or conda) using the requirements.txt
2. Create the file structure mentioned below.
3. Retrieve data from the [FinQA dataset](https://github.com/czyssrs/FinQA)
4. Place data in `data/raw/` (create the file structure)
5. Run `python code/preprocess.py`
6. Run `python code/generate_queries.py`
7. Run `python code/filter_queries.py`

### Model Training & Evaluation

1. Choose parameters inside `code/train.py`
2. Run `python code/train.py` (requires adequate hardware)
3. Run `python code/eval.py` (requires adequate hardware)
4. Results are printed into CLI (or `logs/`)

### Using SLURM

The code has been largely made for use with SLURM on [Snellius](https://www.surf.nl).
However, it is still recommended to have a CUDA GPU locally (should still support TF32 and BF16).
See the `scripts/` folder for SLURM related scripts.

## File Structure

The code expects the following folder structure:

<pre>
bachelor-thesis/
│
├── code/                               # Source code
│   ├── archive/                        # Old legacy code (may be broken)
│   ├── config.py                       # Configurations (filepaths, data settings)
│   ├── create_docid.py                 # Create the docid using keyword extraction
│   ├── eval.py                         # Evaluate the model performance
│   ├── filter_queries.py               # Filter the pseudo-queries
│   ├── generate_queries.py             # Generate pseudo-queries
│   ├── metrics.py                      # Custom metrics and custom loss
│   ├── model.py                        # Model and trainer
│   ├── plot.ipynb                      # Plots
│   ├── preprocess.py                   # Preprocess data from raw FinQA
│   ├── process_data.py                 # Process data for model
│   ├── test_bm25.ipynb                 # Test BM25
│   ├── train.py                        # Train the model
│   └── utils.py                        # Utility functions
│
├── data/                               # Data folder
│   ├── processed/                      # Processed data
│   │   ├── documents_aug.csv           # Corpus with filtered pseudo-queries (Used for training)
│   │   ├── documents_pseudo.csv        # Corpus with pseudo-queries
│   │   ├── documents.csv               # Extracted corpus
│   │   ├── eval.csv                    # Evaluation split
│   │   ├── test.csv                    # Test split
│   │   └── train.csv                   # Training split
│   │
│   └── raw/                            # Raw data from FinQA
│       ├── dev.json                    # Evaluation split
│       ├── private_test.json           # Unused
│       ├── test.json                   # Test split
│       └── train.json                  # Training split
│
├── figures/                            # Figures
│
├── logs/                               # Output logs (mostly for debugging)
│
├── models/                             # Model folder (backbone models, trained models, checkpoints)
│
├── scripts/                            # Bash scripts for SLURM on Snellius
│   ├── slurm/                          # SLURM output logs
│   ├── eval.sh                         # Script for evaluation
│   ├── init_git.sh                     # Initializes ssh key agent
│   ├── init_job.sh                     # Initializes SLURM job script
│   ├── signal_utils.sh                 # Utility script
│   └── train.sh                        # Script for training
│
├── .gitignore                          # Git ignore rules
├── README.md                           # Project overview
└── requirements.txt                    # Python dependencies
</pre>
