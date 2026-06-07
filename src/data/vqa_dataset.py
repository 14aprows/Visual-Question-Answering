from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from src.data.image_preprocessing import load_image, resolve_image_path
from src.data.text_preprocessing import encode_answer,encode_question, normalize_text

class VQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        answer_to_idx,
        word_to_idx,
        transform=None,
        max_question_len=32,
        image_column="image",
        question_column="question",
        answer_column="answer",
    ):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.answer_to_idx = answer_to_idx
        self.word_to_idx = word_to_idx
        self.transform = transform
        self.max_question_len = max_question_len
        self.image_column = image_column
        self.question_column = question_column
        self.answer_column = answer_column

        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.dropna(
            subset=[self.image_column, self.question_column, self.answer_column]
        ).reset_index(drop=True)

        self.df[self.question_column] = self.df[self.question_column].apply(normalize_text)
        self.df[self.answer_column] = self.df[self.answer_column].apply(normalize_text)
        self.df = self.df[self.df[self.answer_column].isin(self.answer_to_idx.keys())].reset_index(drop=True)        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(self.image_dir, row[self.image_column])
        image = load_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        question = encode_question(row[self.question_column], self.word_to_idx, self.max_question_len)
        answer = encode_answer(row[self.answer_column], self.answer_to_idx)
        return {
            "image": image,
            "question": question,
            "answer": answer,
        }