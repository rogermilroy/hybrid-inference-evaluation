import math
from time import time

import torch
from src.data.synthetic_position_dataloader import get_dataloaders
from src.models.hybrid_inference_model import KalmanHybridInference
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm
from src.training.loss import weighted_mse_loss

from src.training.evaluate_model import evaluate_model, evaluate_model_input


def train_one_epoch_input(model, loader, optimizer, criterion, device, weighted, validate_every,
                          val_loader, log_file):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    for num, (obs, states, inputs) in enumerate(loader):
        obs, states, inputs = obs.to(device), states.to(device), inputs.to(device)

        # zero the optimizers gradients from the previous iteration.
        optimizer.zero_grad()

        # compute the prediction.
        # print(obs.shape)
        out, out_list = model(obs, inputs)

        # compute the loss
        if weighted:
            loss = criterion(out_list, states)
        else:
            loss = criterion(out.permute(0, 2, 1), states)

        # propagate the loss back through the network.
        loss.backward()

        # update the weights.
        optimizer.step()

        # add to the epochs loss
        losses.append(float(loss))
        epoch_loss += float(loss)

        # in epoch validation
        if num % validate_every == 0:
            # if we are validating then do that and print the results
            _, val_av_loss = evaluate_model_input(model=model, loader=val_loader,
                                                       criterion=mse_loss,
                                                       device=device)
            print("Batch {} avg validation loss: {}".format(num + 1, val_av_loss))
            log_file.write("Batch {} avg validation loss: {}\n".format(num + 1, val_av_loss))


    return epoch_loss, losses


def train_one_epoch(model, loader, optimizer, criterion, device, weighted, validate_every,
    val_loader, log_file, save_path, early_stopping):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    best_val = math.inf
    best_model = None
    for num, (obs, states) in tqdm(enumerate(loader)):
        obs, states = obs.to(device), states.to(device)

        # zero the optimizers gradients from the previous iteration.
        optimizer.zero_grad()

        # compute the prediction.
        # print(obs.shape)
        out, out_list = model(obs)

        # compute the loss
        if weighted:
            loss = criterion(out_list, states)
        else:
            loss = criterion(out.permute(0, 2, 1), states)

        # propagate the loss back through the network.
        loss.backward()

        # update the weights.
        optimizer.step()

        # add to the epochs loss
        losses.append(float(loss))
        epoch_loss += float(loss)

        # in epoch validation
        if num % validate_every == 0:
            print("validating")
            # if we are validating then do that and print the results
            _, val_av_loss = evaluate_model(model=model, loader=val_loader,
                                                       criterion=mse_loss,
                                                       device=device)

            print("Batch {} avg validation loss: {}".format(num + 1, val_av_loss))
            log_file.write("Batch {} avg validation loss: {}\n".format(num + 1, val_av_loss))
            torch.save(model.state_dict(), save_path + str(num) + ".pt")

            # store the best model based on validation scores.
            if early_stopping and val_av_loss < best_val:
                best_val = val_av_loss
                best_model = model.state_dict()
                # save the model at this point.
                torch.save(model.state_dict(), save_path + "best.pt")

    return epoch_loss, losses, best_model


def train_hybrid_inference(epochs, loss, weighted, save_path, inputs, validate_every,
                           log_path="./training.txt",
                           vis_examples=0,
                           data_params={}, load_model=None, computing_device=torch.device("cpu"),
                           early_stopping=True, gamma=0.005, lr=0.001):

    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]], device=computing_device)
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]], device=computing_device)
    Q = torch.tensor([[0.25 ** 2, 0., 0., 0.],
                      [0., 0.25 ** 2, 0., 0.],
                      [0., 0., 0.25 ** 2, 0.],
                      [0., 0., 0., 0.25 ** 2]], device=computing_device)
    R = (0.35 ** 2) * torch.eye(2, device=computing_device)
    if inputs:
        G = torch.tensor([[1 / 2, 1, 0., 0.],
                          [0., 0., 1 / 2, 1]], device=computing_device).t()
        model = KalmanHybridInference(F=F,
                                      H=H,
                                      Q=Q,
                                      R=R,
                                      G=G,
                                      gamma=gamma)
    else:
        model = KalmanHybridInference(F=F,
                                      H=H,
                                      Q=Q,
                                      R=R,
                                      gamma=gamma)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    model.to(computing_device)

    print("Model on CUDA?", next(model.parameters()).is_cuda)

    criterion = loss
    optimizer = Adam(model.parameters(), lr=lr)

    if data_params:
        # if we have specified specific training parameters use them
        train_samples = data_params["train_samples"]
        val_samples = data_params["val_samples"]
        test_samples = data_params["test_samples"]
        sample_length = data_params["sample_length"]
        starting_point = data_params["starting_point"]
        batch_size = data_params["batch_size"]
        extras = data_params["extras"]
        train_loader, val_loader, test_loader = get_dataloaders(train_samples=train_samples,
                                                                val_samples=val_samples,
                                                                test_samples=test_samples,
                                                                sample_length=sample_length,
                                                                starting_point=starting_point,
                                                                batch_size=batch_size,
                                                                inputs=inputs,
                                                                extras=extras)
    else:
        # else use the defaults.
        train_loader, val_loader, test_loader = get_dataloaders(inputs=inputs)

    divisor = train_loader.dataset.total_samples()

    if inputs:
        _, pre_val = evaluate_model_input(model, test_loader, criterion=mse_loss,
                                       device=computing_device)
    else:
        _, pre_val = evaluate_model(model, test_loader, criterion=mse_loss, device=computing_device)

    best_model = None
    with open(log_path, 'w+') as log_file:

        print("Before training avg test loss: {}".format(pre_val))
        log_file.write("Before training avg test loss: {}".format(pre_val))

        start = time()
        for i in range(epochs):
            # train a simple epoch and record and print the losses.
            if inputs:
                epoch_loss, epoch_losses = train_one_epoch_input(model=model, loader=train_loader,
                                                                 optimizer=optimizer,
                                                                 criterion=criterion,
                                                                 device=computing_device,
                                                                 weighted=weighted,
                                                                 validate_every=validate_every,
                                                                 val_loader=val_loader,
                                                                 log_file=log_file)
            else:
                epoch_loss, epoch_losses, best_model = train_one_epoch(model=model,
                                                                       loader=train_loader,
                                                                       optimizer=optimizer,
                                                                       criterion=criterion,
                                                                       device=computing_device,
                                                                       weighted=weighted,
                                                                       validate_every=validate_every,
                                                                       val_loader=val_loader,
                                                                       log_file=log_file,
                                                                       save_path=save_path,
                                                                       early_stopping=early_stopping)
            print("Epoch {} avg training loss: {}".format(i + 1, epoch_loss / divisor))
            log_file.write("Epoch {} avg training loss: {}\n".format(i + 1, epoch_loss / divisor))

        # load the best model based on early stopping
        if early_stopping:
            model.load_state_dict(best_model)

        # test it.
        if inputs:
            test_loss, test_av_loss = evaluate_model_input(model=model,
                                                           loader=test_loader,
                                                           criterion=mse_loss,
                                                           device=computing_device,
                                                           vis_example=vis_examples)
        else:
            test_loss, test_av_loss = evaluate_model(model=model,
                                                     loader=test_loader,
                                                     criterion=mse_loss,
                                                     device=computing_device,
                                                     vis_example=vis_examples)
        print("Time taken: {}".format(time() - start))
        print("Test average loss: {}".format(test_av_loss))
        log_file.write("Time taken: {}\n".format(time() - start))
        log_file.write("Test average loss: {}\n".format(test_av_loss))


if __name__ == '__main__':

    data = {'train_samples': 100, 'val_samples': 20, 'test_samples': 50, 'sample_length': 100,
            'starting_point': 0, 'batch_size': 1, 'extras': False}

    train_hybrid_inference(epochs=1, loss=mse_loss, weighted=False,
                           save_path='./linear_hi_300', inputs=False, validate_every=10,
                           gamma=0.005,
                           lr=0.0001, data_params=data)
