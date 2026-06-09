import torch 
import torch.optim as optim
from src.engine.trainer import train_one_epoch, validate_one_epoch
from src.data.vqa_dataloader import get_vqa_dataloader
from src.models.vqa_resnet_model import VQAResNetModel
from src.configs.config import LEARNING_RATE, EPOCHS, WEIGHT_DECAY, LABEL_SMOOTHING, GRAD_CLIP_NORM, LOG_DIR, CHECKPOINT_DIR
from src.utils.csv_logger import CSVLogger, get_run_paths

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, answer_to_idx, idx_to_answer, word_to_idx, idx_to_word = get_vqa_dataloader()
    model = VQAResNetModel(
        vocab_size=len(word_to_idx),
        num_answers=len(answer_to_idx),
        freeze_backbone=True,
        train_last_block=True,
    ).to(device)

    run_paths = get_run_paths(
        run_name="vqa_resnet_gru_transfer",
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    run_paths["best_checkpoint"].parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    logger = CSVLogger(
        log_path=run_paths["log_csv"],
        fieldnames=["epoch", "lr", "train_loss", "train_acc","val_loss", "val_acc", "val_top5_acc", "is_best"],
    )

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
    
        val_loss, val_acc, val_top5_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
    
        scheduler.step(val_loss)
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_top5_acc": val_top5_acc,
                    "answer_to_idx": answer_to_idx,
                    "idx_to_answer": idx_to_answer,
                    "word_to_idx": word_to_idx,
                    "idx_to_word": idx_to_word,
                },
                run_paths["best_checkpoint"],
            )
        logger.log(
            {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_top5_acc": val_top5_acc,
                "is_best": int(is_best),
            }
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Top5 Acc: {val_top5_acc:.4f} | "
            f"Best: {is_best}"
        )

if __name__ == "__main__":
    main()
