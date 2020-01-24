from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataloader import get_dataloaders
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm
from .evaluate_model import evaluate_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    for obs, states in loader:
        obs, states = obs.to(device).squeeze(), states.to(device).squeeze()

        # zero the optimizers gradients from the previous iteration.
        optimizer.zero_grad()

        # compute the prediction.
        # print(obs.shape)
        out = model(obs)

        # compute the loss
        loss = criterion(out.t(), states)

        # propagate the loss back through the network.
        loss.backward()

        # update the weights.
        optimizer.step()

        # add to the epochs loss
        losses.append(float(loss))
        epoch_loss += float(loss)

    return epoch_loss, losses


def train_hybrid_inference(epochs, val, loss, save_path, log_path="./training.txt", vis_examples=0,
                           data_params={}, load_model=None, computing_device=torch.device("cpu")):

    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]]).to(computing_device)
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]]).to(computing_device)
    Q = torch.tensor([[0.05 ** 2, 0., 0., 0.],
                      [0., 0.05 ** 2, 0., 0.],
                      [0., 0., 0.05 ** 2, 0.],
                      [0., 0., 0., 0.05 ** 2]]).to(computing_device)
    R = (0.05 ** 2) * torch.eye(2).to(computing_device)
    model = HybridInference(F=F,
                            H=H,
                            Q=Q,
                            R=R,
                            gamma=1e-4)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    model.to(computing_device)

    print("Model on CUDA?", next(model.parameters()).is_cuda)

    criterion = loss
    optimizer = Adam(model.parameters())

    if data_params:
        # if we have specified specific training parameters use them
        train_samples = data_params["train_samples"]
        val_samples = data_params["val_samples"]
        test_samples = data_params["test_samples"]
        sample_length = data_params["sample_length"]
        starting_point = data_params["starting_point"]
        extras = data_params["extras"]
        train_loader, val_loader, test_loader = get_dataloaders(train_samples=train_samples,
                                                                val_samples=val_samples,
                                                                test_samples=test_samples,
                                                                sample_length=sample_length,
                                                                starting_point=starting_point,
                                                                extras=extras)
    else:
        # else use the defaults.
        train_loader, val_loader, test_loader = get_dataloaders()

    with open(log_path, 'w+') as log_file:
        for i in range(epochs):
            # train a simple epoch and record and print the losses.
            epoch_loss, epoch_losses = train_one_epoch(model=model, loader=train_loader,
                                                       optimizer=optimizer, criterion=criterion,
                                                       device=computing_device)
            print("Epoch {} avg training loss: {}".format(i+1, epoch_loss/len(train_loader)))

            if val:
                # if we are validating then do that and print the results
                val_loss, val_av_loss = evaluate_model(model=model, loader=val_loader,
                                                       criterion=criterion, device=computing_device)
                print("Epoch {} validation loss: {}".format(i + 1, val_loss))
                print("Epoch {} avg validation loss: {}".format(i + 1, val_av_loss))
            # save the model at this point.
            torch.save(model.state_dict(), save_path)

        # test it.
        test_loss, test_av_loss = evaluate_model(model=model, loader=test_loader, criterion=criterion,
                                                 device=computing_device, vis_example=vis_examples)
        print("Total test loss: {}".format(test_loss))
        print("Test average loss: {}".format(test_av_loss))
