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
    for obs, states in tqdm(loader):
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


def train_hybrid_inference(epochs, val, save_path, load_model=None):
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 3, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    F = torch.tensor([[1., 1., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]])
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    Q = torch.tensor([[0.05 ** 2, 0., 0., 0.],
                      [0., 0.05 ** 2, 0., 0.],
                      [0., 0., 0.05 ** 2, 0.],
                      [0., 0., 0., 0.05 ** 2]])
    R = (0.05 ** 2) * torch.eye(2)
    model = HybridInference(F=F,
                            H=H,
                            Q=Q,
                            R=R,
                            gamma=1e-4)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    criterion = mse_loss
    optimizer = Adam(model.parameters())
    train_loader, val_loader, test_loader = get_dataloaders()

    for i in range(epochs):
        epoch_loss, epoch_losses = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion,
                        device=computing_device)
        print("Epoch {} avg training loss: {}".format(i+1, epoch_loss/len(train_loader)))
        if val:
            val_loss, val_av_loss = evaluate_model(model=model, loader=val_loader, criterion=criterion, device=computing_device)
            print("Epoch {} validation loss: {}".format(i + 1, val_loss))
            print("Epoch {} avg validation loss: {}".format(i + 1, val_av_loss))
        torch.save(model.state_dict(), save_path)

    # test it.
    test_loss, test_av_loss = evaluate_model(model=model, loader=test_loader, criterion=criterion, device=computing_device)
    print("Total test loss: {}".format(test_loss))
    print("Test average loss: {}".format(test_av_loss))
