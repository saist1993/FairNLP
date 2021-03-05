import time
import torch
from tqdm.auto import tqdm

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, device, accuracy_calculation_function):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for labels, text, lengths in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths)

        loss = criterion(predictions, labels)

        acc = accuracy_calculation_function(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, accuracy_calculation_function):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for labels, text, lengths in iterator:
            labels = labels.to(device)
            text = text.to(device)

            predictions = model(text, lengths)

            loss = criterion(predictions, labels)

            acc = accuracy_calculation_function(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def basic_training_loop(
        n_epochs:int,
        model,
        train_iterator,
        dev_iterator,
        optimizer,
        criterion,
        device,
        model_save_name,
        accuracy_calculation_function
):
    best_valid_loss = float('inf')
    best_valid_acc = float('inf')
    best_test_acc = float('inf')
    test_acc_at_best_valid_acc = float('inf')
    best_valid_acc_epoch = 0


    for epoch in range(n_epochs):

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device, accuracy_calculation_function)
        valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, device, accuracy_calculation_function)
        test_loss, test_acc = evaluate(model, dev_iterator, criterion, device, accuracy_calculation_function)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_name)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_acc_at_best_valid_acc = test_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')

    return best_test_acc, best_valid_acc, test_acc_at_best_valid_acc