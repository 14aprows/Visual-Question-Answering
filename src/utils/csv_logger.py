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

def get_run_paths(run_name, log_dir, checkpoint_dir):
    run_name = str(run_name).lower()
    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)

    return {
        "log_csv": log_dir / f"{run_name}.csv",
        "best_checkpoint": checkpoint_dir / f"{run_name}_best.pt",
        "last_checkpoint": checkpoint_dir / f"{run_name}_last.pt"
    }