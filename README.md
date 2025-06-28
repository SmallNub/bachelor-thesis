# The Integration of Chain-of-Thought in Generative Information Retrieval

This repository hosts the source code used by the bachelor thesis: "The Integration of Chain-of-Thought in Generative Information Retrieval".

## Abstract

Traditional information retrieval (IR) systems follow a multi-stage "index-retrieve-rerank" pipeline, which is storage-intensive, lacks end-to-end optimization, and scales poorly with large corpora. Generative retrieval (GR) offers an alternative by encoding the corpus into model parameters and generating document identifiers directly, but existing GR models lack robust reasoning capabilities.
This research investigates whether incorporating Chain-of-Thought (CoT) prompting can enhance the reasoning capabilities of GR models. A FLAN-T5 model is adapted to retrieve document identifiers for complex queries, with CoT prompting applied during training and inference.
Additionally, a custom loss function is created to enhance the training of CoT-based models.
Furthermore, parameter-efficient methods are explored to reduce training costs.
Experiments conducted on the FinQA dataset, which contains multi-hop financial reasoning questions, demonstrate that models with CoT prompting outperform baselines in both retrieval accuracy and structured answer consistency.
These findings suggest that reasoning-enhanced prompting can meaningfully improve the ability to handle complex queries for GR models.

For more information, see the published paper.

If there are any questions, feel free to open an issue.

## Requirements

- Sufficient python knowledge
- CUDA Hardware that can handle TF32 and BF16
- Internet connection

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
4. Results are printed into cli (or `logs/`)

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
