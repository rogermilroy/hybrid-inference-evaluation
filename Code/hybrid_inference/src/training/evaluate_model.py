from tqdm import tqdm
import torch


def evaluate_model(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.
    with torch.no_grad():
        for obs, states in tqdm(loader):
            obs, states = obs.to(device).squeeze(), states.to(device).squeeze()

            # compute the prediction.
            # print(obs.shape)
            out = model(obs)

            loss = criterion(out, states)

            # add to the epochs loss
            epoch_loss += float(loss)

    return epoch_loss, epoch_loss / len(loader)
