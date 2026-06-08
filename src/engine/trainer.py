import torch
from src.metrics.vqa_metrics import AverageMeter, top_k_accuracy, accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch in train_loader:
        images = batch["image"].to(device)
        questions = batch["question"].to(device)
        answers = batch["answer"].to(device)

        optimizer.zero_grad()

        logits = model(images, questions)
        loss = criterion(logits, answers)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, answers), batch_size)

    avg_loss = loss_meter.avg
    acc = acc_meter.avg
    return avg_loss, acc

@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    top5_meter = AverageMeter()

    for batch in val_loader:
        images = batch["image"].to(device)
        questions = batch["question"].to(device)
        answers = batch["answer"].to(device)

        logits = model(images, questions)
        loss = criterion(logits, answers)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, answers), batch_size)
        top5_meter.update(top_k_accuracy(logits, answers, k=5), batch_size)

    avg_loss = loss_meter.avg
    acc = acc_meter.avg
    top5_acc = top5_meter.avg
    return avg_loss, acc, top5_acc
