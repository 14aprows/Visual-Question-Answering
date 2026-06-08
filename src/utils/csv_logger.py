import csv 
from pathlib import Path

class CSVLogger:
    def __init__(self, log_path, fieldnames):
        self.log_path = Path(log_path)
        self.fieldnames = fieldnames
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
    def log(self, row):
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key) for key in self.fieldnames})