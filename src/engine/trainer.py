import torch

def train_one_epoch(model, train_loader, criterion, optimizer, device):    
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        questions = batch['question'].to(device)
        answers = batch['answer'].to(device)

        optimizer.zero_grad()

        logits = model(images, questions)
        loss = criterion(logits, answers)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (logits.argmax(1) == answers).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        images = batch['image'].to(device)
        questions = batch['question'].to(device)
        answers = batch['answer'].to(device)

        logits = model(images, questions)
        loss = criterion(logits, answers)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        total_correct += (logits.argmax(1) == answers).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
