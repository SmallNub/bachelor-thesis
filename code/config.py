# DATA PATHS

# Directories
DATA_DIR = "data"
DATA_DIR_RAW = DATA_DIR + "/raw"
DATA_DIR_PROC = DATA_DIR + "/processed"

# Train
DATA_TRAIN_RAW = DATA_DIR_RAW + "/train.json"
DATA_TRAIN_PROC = DATA_DIR_PROC + "/train.csv"

# Eval
DATA_EVAL_RAW = DATA_DIR_RAW + "/dev.json"
DATA_EVAL_PROC = DATA_DIR_PROC + "/eval.csv"

# Test
DATA_TEST_RAW = DATA_DIR_RAW + "/test.json"
DATA_TEST_PROC = DATA_DIR_PROC + "/test.csv"

# Documents
DATA_DOCUMENTS = DATA_DIR_PROC + "/documents.csv"
DATA_DOCUMENTS_PSEUDO = DATA_DIR_PROC + "/documents_pseudo.csv"
DATA_DOCUMENTS_AUG = DATA_DIR_PROC + "/documents_aug.csv"


# MODEL PATHS

MODELS_DIR = "models"


# CONFIG

SPLITS = ["train", "eval", "test"]

# Separator used for parts inside the docid
SEPARATOR = "-"

# Keyword extraction
KEYWORDS_NGRAM_SIZE = 2  # Hard-coded for CoT
KEYWORDS_DIVERSITY = 0.5
KEYWORDS_TOP_N = 2  # Hard-coded for CoT

# Docid creation
USE_COMPANY_YEAR_IN_DOCID = True
DOCID_SIZE = KEYWORDS_NGRAM_SIZE * KEYWORDS_TOP_N + (2 if USE_COMPANY_YEAR_IN_DOCID else 0)
