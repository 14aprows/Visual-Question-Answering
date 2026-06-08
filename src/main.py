import torch 
import torch.optim as optim
from src.engine.trainer import train_one_epoch, validate_one_epoch
from src.data.vqa_dataloader import get_vqa_dataloader
from src.models.vqa_model import VQAModel
from src.configs.config import LEARNING_RATE, EPOCHS

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, answer_to_idx, idx_to_answer, word_to_idx, idx_to_word = get_vqa_dataloader()
    model = VQAModel(
        vocab_size=len(word_to_idx),
        num_answers=len(answer_to_idx),
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_top5_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step(val_loss)
        
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Top5 Acc: {val_top5_acc:.4f}"
        )
        
if __name__ == "__main__":
    main()