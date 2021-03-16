import time
import torch
from tqdm.auto import tqdm

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, device, accuracy_calculation_function, other_params):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    is_regression = other_params['is_regression']

    for labels, text, lengths in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)

        optimizer.zero_grad()

        predictions = model(text, lengths)

        if is_regression:
            loss = criterion(predictions.squeeze(), labels.squeeze())
        else:
            loss = criterion(predictions, labels)


        acc = accuracy_calculation_function(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, accuracy_calculation_function, other_params):
    epoch_loss = 0
    epoch_acc = 0

    is_regression = other_params['is_regression']
    model.eval()

    with torch.no_grad():
        for labels, text, lengths in iterator:
            labels = labels.to(device)
            text = text.to(device)

            predictions = model(text, lengths)

            # loss = criterion(predictions, labels)
            if is_regression:
                loss = criterion(predictions.squeeze(), labels.squeeze())
            else:
                loss = criterion(predictions, labels)

            acc = accuracy_calculation_function(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_adv(model, iterator, optimizer, criterion, device, accuracy_calculation_function, other_params):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    loss_aux_scale = other_params["loss_aux_scale"]
    is_regression = other_params['is_regression']
    is_post_hoc = other_params['is_post_hoc']

    for labels, text, lengths, aux in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)
        aux = aux.to(device)

        optimizer.zero_grad()



        # loss = criterion(predictions, labels)

        if is_post_hoc:
            predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), aux.squeeze())
                # loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                # loss_main = criterion(predictions, labels)
                loss_main = criterion(predictions, aux)
            acc = accuracy_calculation_function(predictions, aux)
            loss_aux = 0.0
        else:
            predictions, aux_predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_main = criterion(predictions, labels)
                loss_aux = criterion(aux_predictions, aux)
            acc = accuracy_calculation_function(predictions, labels)

        loss = loss_main + (loss_aux_scale*loss_aux)



        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate_adv(model, iterator, criterion, device, accuracy_calculation_function, other_params):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    loss_aux_scale = other_params["loss_aux_scale"]
    is_regression = other_params['is_regression']
    is_post_hoc = other_params['is_post_hoc']
    all_predictions = []

    with torch.no_grad():
        for labels, text, lengths, aux in tqdm(iterator):
            labels = labels.to(device)
            text = text.to(device)
            aux = aux.to(device)

            if is_post_hoc:
                predictions = model(text, lengths)
                if is_regression:
                    loss_main = criterion(predictions.squeeze(), aux.squeeze())
                    # loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
                else:
                    # loss_main = criterion(predictions, labels)
                    loss_main = criterion(predictions, aux)
                acc = accuracy_calculation_function(predictions, aux)
                loss_aux = 0.0
            else:
                predictions, aux_predictions = model(text, lengths)
                if is_regression:
                    loss_main = criterion(predictions.squeeze(), labels.squeeze())
                    loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
                else:
                    loss_main = criterion(predictions, labels)
                    loss_aux = criterion(aux_predictions, aux)
                acc = accuracy_calculation_function(predictions, labels)

            loss = loss_main + (loss_aux_scale * loss_aux)
            # all_predictions.append(aux_predictions.squeeze(),labels, aux.squeeze(), )
            # acc = accuracy_calculation_function(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def basic_training_loop(
        n_epochs:int,
        model,
        train_iterator,
        dev_iterator,
        test_iterator,
        optimizer,
        criterion,
        device,
        model_save_name,
        accuracy_calculation_function,
        wandb,
        other_params = {'is_adv': False}
):

    best_valid_loss = 1*float('inf')
    best_valid_acc = -1*float('inf')
    best_test_acc = -1*float('inf')
    test_acc_at_best_valid_acc = -1*float('inf')
    best_valid_acc_epoch = 0
    is_adv = other_params['is_adv']
    save_model = other_params['save_model']
    print(f"is adv: {is_adv}")

    for epoch in range(n_epochs):

        start_time = time.monotonic()

        if is_adv:
            train_loss, train_acc = train_adv(model, train_iterator, optimizer, criterion, device,
                                          accuracy_calculation_function, other_params)
            valid_loss, valid_acc = evaluate_adv(model, dev_iterator, criterion, device, accuracy_calculation_function,
                                             other_params)
            test_loss, test_acc = evaluate_adv(model, test_iterator, criterion, device, accuracy_calculation_function,
                                           other_params)
        else:
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device, accuracy_calculation_function, other_params)
            valid_loss, valid_acc = evaluate(model, dev_iterator, criterion, device, accuracy_calculation_function, other_params)
            test_loss, test_acc = evaluate(model, test_iterator, criterion, device, accuracy_calculation_function, other_params)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if save_model:
            if valid_loss < best_valid_loss:
                print(f"model saved as: {model_save_name}")
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_name)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            test_acc_at_best_valid_acc = test_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Val. Acc: {test_acc}%')

        if wandb:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'epoch': epoch,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'test_acc': test_acc
            })

    return best_test_acc, best_valid_acc, test_acc_at_best_valid_acc