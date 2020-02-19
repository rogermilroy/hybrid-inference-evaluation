from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataloader import get_dataloaders
import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss
from time import time
from tqdm import tqdm
from .evaluate_model import evaluate_model, evaluate_model_input


def train_one_epoch_input(model, loader, optimizer, criterion, device, weighted):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    for obs, states, inputs in loader:
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

    return epoch_loss, losses


def train_one_epoch(model, loader, optimizer, criterion, device, weighted):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    for obs, states in loader:
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

    return epoch_loss, losses


def train_hybrid_inference(epochs, val, loss, weighted, save_path, inputs,
                           log_path="./training.txt",
                           vis_examples=0,
                           data_params={}, load_model=None, computing_device=torch.device("cpu")):

    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]], device=computing_device)
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]], device=computing_device)
    Q = torch.tensor([[0.15 ** 2, 0., 0., 0.],
                      [0., 0.15 ** 2, 0., 0.],
                      [0., 0., 0.15 ** 2, 0.],
                      [0., 0., 0., 0.15 ** 2]], device=computing_device)
    R = (0.25 ** 2) * torch.eye(2, device=computing_device)
    if inputs:
        G = torch.tensor([[1 / 2, 1, 0., 0.],
                         [0., 0., 1 / 2, 1]], device=computing_device).t()
        model = HybridInference(F=F,
                                H=H,
                                Q=Q,
                                R=R,
                                G=G,
                                gamma=1e-3)
    else:
        model = HybridInference(F=F,
                                H=H,
                                Q=Q,
                                R=R,
                                gamma=1e-3)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    model.to(computing_device)

    print("Model on CUDA?", next(model.parameters()).is_cuda)

    criterion = loss
    optimizer = Adam(model.parameters(), lr=0.0005)  # reduce the learning rate to 0.0001 from 0.001

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

    with open(log_path, 'w+') as log_file:
        start = time()
        for i in range(epochs):
            # train a simple epoch and record and print the losses.
            if inputs:
                epoch_loss, epoch_losses = train_one_epoch_input(model=model, loader=train_loader,
                                                           optimizer=optimizer, criterion=criterion,
                                                           device=computing_device, weighted=weighted)
            else:
                epoch_loss, epoch_losses = train_one_epoch(model=model, loader=train_loader,
                                                                 optimizer=optimizer,
                                                                 criterion=criterion,
                                                                 device=computing_device,
                                                                 weighted=weighted)
            print("Epoch {} avg training loss: {}".format(i+1, epoch_loss/len(train_loader)))
            log_file.write("Epoch {} avg training loss: {}\n".format(i + 1, epoch_loss / len(
                train_loader)))

            if val:
                # if we are validating then do that and print the results
                if inputs:
                    val_loss, val_av_loss = evaluate_model_input(model=model, loader=val_loader,
                                                       criterion=mse_loss, device=computing_device)
                else:
                    val_loss, val_av_loss = evaluate_model(model=model, loader=val_loader,
                                                           criterion=mse_loss,
                                                           device=computing_device)
                print("Epoch {} validation loss: {}".format(i + 1, val_loss))
                print("Epoch {} avg validation loss: {}".format(i + 1, val_av_loss))
                log_file.write("Epoch {} validation loss: {}\n".format(i + 1, val_loss))
                log_file.write("Epoch {} avg validation loss: {}\n".format(i + 1, val_av_loss))
            # save the model at this point.
            torch.save(model.state_dict(), save_path)

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
        print("Total test loss: {}".format(test_loss))
        print("Test average loss: {}".format(test_av_loss))
        log_file.write("Time taken: {}\n".format(time() - start))
        log_file.write("Total test loss: {}\n".format(test_loss))
        log_file.write("Test average loss: {}\n".format(test_av_loss))
