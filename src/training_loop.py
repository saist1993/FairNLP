import time
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union
from models import initialize_parameters

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
    '''
        ADV: return adv loss, adv acc, main acc, main loss
    '''

    epoch_loss_main = []
    epoch_loss_aux = []
    epoch_acc_aux = []
    epoch_acc_main = []
    epoch_total_loss = []

    model.train()

    loss_aux_scale = other_params["loss_aux_scale"]
    is_regression = other_params['is_regression']
    is_post_hoc = other_params['is_post_hoc']


    for labels, text, lengths, aux in tqdm(iterator):

        labels = labels.to(device)
        text = text.to(device)
        aux = aux.to(device)
        optimizer.zero_grad()

        if is_post_hoc:
            predictions = model(text, lengths)
            if is_regression:
                loss_aux = criterion(predictions.squeeze(), aux.squeeze())
                # loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                # loss_main = criterion(predictions, labels)
                loss_aux = criterion(predictions, aux)

            loss_aux.backward()
            acc_aux = accuracy_calculation_function(predictions, aux)
            acc_main = torch.zeros(1)
            loss_main = torch.zeros(1)
            total_loss = torch.zeros(1)

        else:
            predictions, aux_predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_main = criterion(predictions, labels)
                loss_aux = criterion(aux_predictions, aux)

            acc_main = accuracy_calculation_function(predictions, labels)
            acc_aux = accuracy_calculation_function(aux_predictions, aux)


            total_loss = loss_main + (loss_aux_scale*loss_aux)
            total_loss.backward()

        optimizer.step()

        epoch_loss_main.append(loss_main.item())
        epoch_acc_main.append(acc_main.item())
        epoch_loss_aux.append(loss_aux.item())
        epoch_acc_aux.append(acc_aux.item())
        epoch_total_loss.append(total_loss.item())

    return np.mean(epoch_total_loss), np.mean(epoch_loss_main), np.mean(epoch_acc_main), np.mean(epoch_loss_aux), np.mean(epoch_acc_aux)


def train_adv_three_phase(model, iterator, optimizer, criterion, device, accuracy_calculation_function, phase, other_params):

    model.train()
    is_regression = other_params['is_regression']
    loss_aux_scale = other_params["loss_aux_scale"]

    epoch_loss_main = 0
    epoch_acc_main = 0
    epoch_loss_aux = 0
    epoch_acc_aux = 0
    print(phase)

    for labels, text, lengths, aux in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)
        aux = aux.to(device)

        if phase == 'initial':
            """
            initial phase:
                Train Embedder + Classifier for one batch
                Train Freeze(Embedder) + Adv for one batch
            """
            print("inside initial phase")

            optimizer.zero_grad()
            predictions, aux_predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
            else:
                loss_main = criterion(predictions, labels)

            loss_main.backward()
            optimizer.step()

        elif phase == 'perturbate' or phase == 'recover':
            optimizer.zero_grad()
            model.freeze_unfreeze_embedder(freeze=True)

            predictions, aux_predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
            else:
                loss_main = criterion(predictions, labels)

            loss_main.backward()
            optimizer.step()
            model.freeze_unfreeze_embedder(freeze=False)



        if phase == 'initial' or phase == 'perturbate':
            # optimizer.zero_grad()
            model.freeze_unfreeze_embedder(freeze=True)
            predictions, aux_predictions = model(text, lengths)

            if is_regression:
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_aux = criterion(aux_predictions, aux)

            loss_aux.backward()
            optimizer.step()
            model.freeze_unfreeze_embedder(freeze=False)

        elif phase == 'recover':
            optimizer.zero_grad()
            model.freeze_unfreeze_embedder(freeze=True)
            predictions, aux_predictions = model(text, lengths)

            if is_regression:
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_aux = criterion(aux_predictions, aux)

            loss_aux.backward()
            optimizer.step()
            model.freeze_unfreeze_embedder(freeze=False)

        if phase == 'perturbate':
            model.freeze_unfreeze_classifier(freeze=True)
            model.freeze_unfreeze_adv(freeze=True)
            optimizer.zero_grad()
            predictions, aux_predictions = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_main = criterion(predictions, labels)
                loss_aux = criterion(aux_predictions, aux)
            loss_main = loss_main - (loss_aux_scale * loss_aux)
            loss_main.backward()
            optimizer.step()
            model.freeze_unfreeze_classifier(freeze=False)
            model.freeze_unfreeze_adv(freeze=False)

        if phase != 'recover':
            loss_aux = torch.zeros(1)

        acc_main = accuracy_calculation_function(predictions, labels)
        acc_aux = accuracy_calculation_function(aux_predictions, aux)

        # now we have acc_main, acc_aux, loss_main, loss_aux .. Log this

        epoch_loss_main += loss_main.item()
        epoch_acc_main += acc_main.item()
        epoch_loss_aux += loss_aux.item()
        epoch_acc_aux += acc_aux.item()

    return epoch_loss_main/ len(iterator), epoch_loss_aux/ len(iterator), epoch_acc_main/ len(iterator), epoch_acc_aux/ len(iterator)


def freeze(opt: torch.optim, layer: str, model: torch.nn.Module):
    opt.param_groups[model.legend[layer]]['lr'] = 0

def unfreeze(opt: torch.optim, layer: int, lr, model: torch.nn.Module):
    opt.param_groups[model.legend[layer]]['lr'] = lr

def train_adv_three_phase_custom(model, iterator, optimizer, criterion, device, accuracy_calculation_function, phase, other_params):

    model.train()
    is_regression = other_params['is_regression']
    loss_aux_scale = other_params["loss_aux_scale"]

    epoch_loss_main = 0
    epoch_acc_main = 0
    epoch_loss_aux = 0
    epoch_acc_aux = 0
    # print(phase)

    for labels, text, lengths, aux in tqdm(iterator):
        labels = labels.to(device)
        text = text.to(device)
        aux = aux.to(device)

        if phase == 'initial' or phase == 'recover':
            """
            initial phase:
                Train Embedder + Classifier for one batch
                Train Freeze(Embedder) + Adv for one batch
            recover phase
                Train Freeze (Embedder) + Classifier
                Train Freeze (Embedder) + Adv
            """
            #
            if phase == 'recover':
                freeze(optimizer, model=model, layer='encoder')
                # model.freeze_unfreeze_embedder(freeze=True)

            # --- train Embedder and Classifier
            # model.freeze_unfreeze_adv(freeze=True)
            freeze(optimizer, model=model, layer='adversary')
            optimizer.zero_grad()
            predictions, aux_predictions1 = model(text, lengths)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
            else:
                loss_main = criterion(predictions, labels)
            loss_main.backward()
            optimizer.step()
            # model.freeze_unfreeze_adv(freeze=False)
            unfreeze(optimizer, model=model, layer='adversary', lr=0.01)
            # -- Training ends ---

            # -- Train freeze(E) + Adv
            optimizer.zero_grad()
            # model.freeze_unfreeze_classifier(freeze=True)
            # model.freeze_unfreeze_embedder(freeze=True)
            freeze(optimizer, model=model, layer='encoder')
            freeze(optimizer, model=model, layer='classifier')
            predictions, aux_predictions = model(text, lengths)

            # if phase == 'recover':
            #     if not torch.equal(torch.argmax(aux_predictions1, dim=1) , torch.argmax(aux_predictions, dim=1)):
            #         print("something wrong")

            if is_regression:
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_aux = criterion(aux_predictions, aux)

            loss_aux.backward()
            optimizer.step()
            # model.freeze_unfreeze_embedder(freeze=False)
            # model.freeze_unfreeze_classifier(freeze=False)
            unfreeze(optimizer, model=model, layer='encoder', lr=0.01)
            unfreeze(optimizer, model=model, layer='classifier', lr=0.01)
            # -- Training ends ---


        elif phase == 'perturbate':
            ''' Gradient reversal layer'''
            #
            # model.freeze_unfreeze_embedder(freeze=False)
            # model.freeze_unfreeze_classifier(freeze=False)
            # model.freeze_unfreeze_adv(freeze=False)
            unfreeze(optimizer, model=model, layer='encoder', lr=0.01)
            unfreeze(optimizer, model=model, layer='classifier', lr=0.01)
            unfreeze(optimizer, model=model, layer='adversary', lr=0.01)

            optimizer.zero_grad()

            predictions, aux_predictions = model(text, lengths, gradient_reversal=True)
            if is_regression:
                loss_main = criterion(predictions.squeeze(), labels.squeeze())
                loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
            else:
                loss_main = criterion(predictions, labels)
                loss_aux = criterion(aux_predictions, aux)

            total_loss = loss_main + (loss_aux_scale*loss_aux)
            total_loss.backward()
            optimizer.step()
        #
        # if phase != 'recover':
        #     loss_aux = torch.zeros(1)

        acc_main = accuracy_calculation_function(predictions, labels)
        acc_aux = accuracy_calculation_function(aux_predictions, aux)

        # now we have acc_main, acc_aux, loss_main, loss_aux .. Log this

        epoch_loss_main += loss_main.item()
        epoch_acc_main += acc_main.item()
        epoch_loss_aux += loss_aux.item()
        epoch_acc_aux += acc_aux.item()

    return epoch_loss_main/ len(iterator), epoch_loss_aux/ len(iterator), epoch_acc_main/ len(iterator), epoch_acc_aux/ len(iterator)



def evaluate_adv(model, iterator, criterion, device, accuracy_calculation_function, other_params):

    epoch_loss_main = []
    epoch_loss_aux = []
    epoch_acc_aux = []
    epoch_acc_main = []
    epoch_total_loss = []

    model.eval()
    loss_aux_scale = other_params["loss_aux_scale"]
    is_regression = other_params['is_regression']
    is_post_hoc = other_params['is_post_hoc']



    with torch.no_grad():
        for labels, text, lengths, aux in tqdm(iterator):
            labels = labels.to(device)
            text = text.to(device)
            aux = aux.to(device)

            if is_post_hoc:
                predictions = model(text, lengths)
                if is_regression:
                    loss_aux = criterion(predictions.squeeze(), aux.squeeze())
                    # loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
                else:
                    # loss_main = criterion(predictions, labels)
                    loss_aux = criterion(predictions, aux)

                acc_aux = accuracy_calculation_function(predictions, aux)
                acc_main = torch.zeros(1)
                loss_main = torch.zeros(1)
                total_loss = torch.zeros(1)
            else:
                predictions, aux_predictions = model(text, lengths)
                if is_regression:
                    loss_main = criterion(predictions.squeeze(), labels.squeeze())
                    loss_aux = criterion(aux_predictions.squeeze(), aux.squeeze())
                else:
                    loss_main = criterion(predictions, labels)
                    loss_aux = criterion(aux_predictions, aux)
                acc_main = accuracy_calculation_function(predictions, labels)
                acc_aux = accuracy_calculation_function(aux_predictions, aux)

                total_loss = loss_main + (loss_aux_scale * loss_aux)

            # loss = loss_main + (loss_aux_scale * loss_aux)
            # all_predictions.append(aux_predictions.squeeze(),labels, aux.squeeze(), )
            # acc = accuracy_calculation_function(predictions, labels)

            epoch_loss_main.append(loss_main.item())
            epoch_acc_main.append(acc_main.item())
            epoch_loss_aux.append(loss_aux.item())
            epoch_acc_aux.append(acc_aux.item())
            epoch_total_loss.append(total_loss.item())

    return np.mean(epoch_total_loss ), np.mean(epoch_loss_main), np.mean(epoch_acc_main), np.mean(epoch_loss_aux), np.mean(epoch_acc_aux)



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
    try:
        is_post_hoc = other_params['is_post_hoc']
    except KeyError:
        is_post_hoc = False

    print(f"is adv: {is_adv}")

    for epoch in range(n_epochs):

        start_time = time.monotonic()

        if is_adv:


            train_total_loss, train_loss_main, train_acc_main, train_loss_aux, train_acc_aux = train_adv(model, train_iterator, optimizer, criterion, device,
                                          accuracy_calculation_function, other_params)

            valid_total_loss, valid_loss_main, valid_acc_main, valid_loss_aux, valid_acc_aux = evaluate_adv(model, dev_iterator, criterion, device, accuracy_calculation_function,
                                             other_params)
            test_total_loss, test_loss_main, test_acc_main, test_loss_aux, test_acc_aux = evaluate_adv(model, test_iterator, criterion, device, accuracy_calculation_function,
                                           other_params)

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if is_post_hoc:
                # log stuff there and model is not save in this setting.
                valid_loss, valid_acc = valid_loss_aux, valid_acc_aux
                train_loss, train_acc = train_loss_aux, train_acc_aux
                test_loss, test_acc = test_loss_aux, test_acc_aux


                # check the best accuracy and update it

                # log all the required stuff

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    test_acc_at_best_valid_acc = test_acc

                if test_acc > best_test_acc:
                    best_test_acc = test_acc

                print(f'Posthoc: Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc}%')
                print(f'\t Test Loss: {test_loss:.3f} |  Val. Acc: {test_acc}%')

                if wandb:
                    wandb.log({
                        'post_hoc_train_loss': train_loss,
                        'post_hoc_valid_loss': valid_loss,
                        'post_hoc_test_loss': test_loss,
                        'post_hoc_epoch': epoch,
                        'post_hoc_train_acc': train_acc,
                        'post_hoc_valid_acc': valid_acc,
                        'post_hoc_test_acc': test_acc
                    })

            else:
                # log stuff here
                valid_loss, valid_acc = valid_total_loss, valid_acc_main
                train_loss, train_acc = train_total_loss, train_acc_main
                test_loss, test_acc = test_total_loss, test_acc_main

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
                        'train_loss_total': train_total_loss,
                        'train_loss_main': train_loss_main,
                        'train_loss_aux': train_loss_aux,
                        'valid_loss_total': valid_total_loss,
                        'valid_loss_main': valid_loss_main,
                        'valid_loss_aux': valid_loss_aux,
                        'test_loss_total': test_total_loss,
                        'test_loss_main': test_loss_main,
                        'test_loss_aux': test_loss_aux,
                        'train_acc_main': train_acc_main,
                        'train_acc_aux': train_acc_aux,
                        'valid_acc_main': valid_acc_main,
                        'valid_acc_aux': valid_acc_aux,
                        'test_acc_main': test_acc_main,
                        'test_acc_aux': test_acc_aux,
                        'epoch': epoch
                    })


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


def three_phase_training_loop(
        n_epochs: int,
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
        other_params={'is_adv': False}
):
    best_valid_loss = 1*float('inf')
    best_valid_acc = -1*float('inf')
    best_test_acc = -1*float('inf')
    test_acc_at_best_valid_acc = -1*float('inf')
    best_valid_acc_epoch = 0
    is_adv = other_params['is_adv']
    save_model = other_params['save_model']
    print(f"is adv: {is_adv}")
    try:
        is_post_hoc = other_params['is_post_hoc']
    except KeyError:
        is_post_hoc = False

    assert is_adv == True

    phase = 'initial'
    is_adv_new = False

    for epoch in range(n_epochs):

        if epoch < int(n_epochs*.30):
            phase = 'initial'
        elif epoch >=int(n_epochs*.30) and epoch< int(n_epochs*.60):
            phase = 'perturbate'
        else:
            phase = 'recover'
            if not is_adv_new:
                model.adv.apply(initialize_parameters)
                is_adv_new = True

        print(f"current phase: {phase}")

        start_time = time.monotonic()
        train_loss_main, train_loss_aux, train_acc_main,train_acc_aux  = train_adv_three_phase_custom(model, train_iterator, optimizer, criterion, device,
                                          accuracy_calculation_function, phase, other_params)
        valid_total_loss, valid_loss_main, valid_acc_main, valid_loss_aux, valid_acc_aux = evaluate_adv(model, dev_iterator, criterion, device, accuracy_calculation_function,
                                             other_params)
        test_total_loss, test_loss_main, test_acc_main, test_loss_aux, test_acc_aux = evaluate_adv(model, test_iterator, criterion, device, accuracy_calculation_function,
                                           other_params)

        train_total_loss = 0


        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)


        valid_loss, valid_acc = valid_total_loss, valid_acc_main
        train_loss, train_acc = train_total_loss, train_acc_main
        test_loss, test_acc = test_total_loss, test_acc_main

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
                'train_loss_total': train_total_loss,
                'train_loss_main': train_loss_main,
                'train_loss_aux': train_loss_aux,
                'valid_loss_total': valid_total_loss,
                'valid_loss_main': valid_loss_main,
                'valid_loss_aux': valid_loss_aux,
                'test_loss_total': test_total_loss,
                'test_loss_main': test_loss_main,
                'test_loss_aux': test_loss_aux,
                'train_acc_main': train_acc_main,
                'train_acc_aux': train_acc_aux,
                'valid_acc_main': valid_acc_main,
                'valid_acc_aux': valid_acc_aux,
                'test_acc_main': test_acc_main,
                'test_acc_aux': test_acc_aux,
                'epoch': epoch
            })



    return best_test_acc, best_valid_acc, test_acc_at_best_valid_acc
