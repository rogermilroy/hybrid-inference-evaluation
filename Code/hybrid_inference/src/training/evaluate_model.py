from tqdm import tqdm
import torch
from src.models.hybrid_inference_model import HybridInference


def evaluate_model(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0.
    with torch.no_grad():
        for obs, states in tqdm(loader):
            obs, states = obs.to(device).squeeze(), states.to(device).squeeze()

            # compute the prediction.
            # print(obs.shape)
            out = model(obs)

            loss = criterion(out.t(), states)

            # add to the epochs loss
            epoch_loss += float(loss)

    return epoch_loss, epoch_loss / len(loader)


if __name__ == '__main__':
    model = HybridInference(F=torch.tensor([[1., 1., 0., 0.],
                                            [0., 1., 0., 0.],
                                            [0., 0., 1., 1.],
                                            [0., 0., 0., 1.]]),
                            H=torch.tensor([[1., 0., 0., 0.],
                                            [0., 0., 1., 0.]]),
                            Q=torch.tensor([[0.05 ** 2, 0., 0., 0.],
                                            [0., 0.05 ** 2, 0., 0.],
                                            [0., 0., 0.05 ** 2, 0.],
                                            [0., 0., 0., 0.05 ** 2]]),
                            R=(0.05 ** 2) * torch.eye(2),
                            gamma=1e-4)
    model.load_state_dict(torch.load("./hybrid_inference_params.pt"))

