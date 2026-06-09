from pathlib import Path
import torch 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "images"
TRAIN_CSV = DATASET_DIR / "data_train.csv"
EVAL_CSV = DATASET_DIR / "data_eval.csv"
FULL_CSV = DATASET_DIR / "data.csv"

TRAIN_IMAGES_LIST = DATASET_DIR / "train_images_list.txt"
TEST_IMAGES_LIST = DATASET_DIR / "test_images_list.txt"

ALL_QA_PAIRS = DATASET_DIR / "all_qa_pairs.txt"
ANSWER_SPACE = DATASET_DIR / "answer_space.txt"

LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

IMAGE_SIZE = 224
SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
