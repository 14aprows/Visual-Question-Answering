import torch 
from torch.utils.data import DataLoader
from src.configs.config import TRAIN_CSV, IMAGE_SIZE, IMAGE_DIR, EVAL_CSV, BATCH_SIZE
from src.data.text_preprocessing import build_answer_vocab, build_question_vocab
from src.data.image_preprocessing import get_image_transform
from src.data.vqa_dataset import VQADataset

def get_vqa_dataloader():
    answer_to_idx, idx_to_answer = build_answer_vocab(
        csv_path=TRAIN_CSV,
        answer_column="answer",
        top_k=582
    )

    word_to_idx, idx_to_word = build_question_vocab(
        csv_path=TRAIN_CSV,
        question_column="question",
        min_freq=1
    )

    train_transform = get_image_transform(IMAGE_SIZE)
    val_transform = get_image_transform(IMAGE_SIZE)

    train_dataset = VQADataset(
        csv_path=TRAIN_CSV,
        image_dir=IMAGE_DIR,
        answer_to_idx=answer_to_idx,
        word_to_idx=word_to_idx,
        transform=train_transform,
        max_question_len=24,
        image_column="image_id",
        question_column="question",
        answer_column="answer",
    )

    val_dataset = VQADataset(
        csv_path=EVAL_CSV,
        image_dir=IMAGE_DIR,
        answer_to_idx=answer_to_idx,
        word_to_idx=word_to_idx,
        transform=val_transform,
        max_question_len=24,
        image_column="image_id",
        question_column="question",
        answer_column="answer",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, answer_to_idx, idx_to_answer, word_to_idx, idx_to_word