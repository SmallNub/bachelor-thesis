# DATA PATHS

DATA_DIR = "data"
DATA_DIR_RAW = DATA_DIR + "/raw"
DATA_DIR_PROC = DATA_DIR + "/processed"

DATA_TRAIN_RAW = DATA_DIR_RAW + "/train.json"
DATA_TRAIN_PROC = DATA_DIR_PROC + "/train.csv"

DATA_VALID_RAW = DATA_DIR_RAW + "/dev.json"
DATA_VALID_PROC = DATA_DIR_PROC + "/valid.csv"

DATA_TEST_RAW = DATA_DIR_RAW + "/test.json"
DATA_TEST_PROC = DATA_DIR_PROC + "/test.csv"

DATA_DOCUMENTS = DATA_DIR_PROC + "/documents.csv"

USED_COLUMNS = ["full_text", "table", "id", "question", "answer", "exe_ans", "steps", "program", "program_re"]

# MODEL PATHS

MODELS_DIR = "models"
