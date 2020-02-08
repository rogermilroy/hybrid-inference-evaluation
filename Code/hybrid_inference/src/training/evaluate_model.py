from tqdm import tqdm
import torch
from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataloader import get_dataloaders
from torch.nn.functional import mse_loss


def evaluate_model_input(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states, inputs) in enumerate(loader):
            obs, states, inputs = obs.to(device), states.to(device), inputs.to(device)

            # compute the prediction.
            out, out_list = model(obs, inputs)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(0, 2, 1), states)

            # add to the epochs loss
            epoch_loss += float(loss)

    return epoch_loss, (epoch_loss / len(loader)) / loader.batch_size


def evaluate_model(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states) in enumerate(loader):
            obs, states = obs.to(device).squeeze(), states.to(device).squeeze()

            # compute the prediction.
            out, out_list = model(obs)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(0, 2, 1), states)

            # add to the epochs loss
            epoch_loss += float(loss)

    return epoch_loss, (epoch_loss / len(loader)) / loader.batch_size


def compare_models(path_to_model1: str, path_to_model2: str):
    # need datasets
    train_loader, val_loader, test_loader = get_dataloaders(train_samples=10,
                                                            val_samples=10,
                                                            test_samples=200,
                                                            sample_length=10,
                                                            starting_point=1,
                                                            batch_size=4,
                                                            extras=False)
    # create models
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
    model1 = HybridInference(F=F,
                            H=H,
                            Q=Q,
                            R=R,
                            gamma=1e-4)
    model2 = HybridInference(F=F,
                            H=H,
                            Q=Q,
                            R=R,
                            gamma=1e-4)
    # load parameters
    model1.load_state_dict(torch.load(path_to_model1, map_location=torch.device("cpu")))
    model2.load_state_dict(torch.load(path_to_model2, map_location=torch.device("cpu")))

    # evaluate and print/return results.
    loss1, av_loss1 = evaluate_model(model=model1, loader=test_loader, criterion=mse_loss,
                                     weighted=False, device=torch.device('cpu'))
    loss2, av_loss2 = evaluate_model(model=model2, loader=test_loader, criterion=mse_loss,
                                     weighted=False, device=torch.device('cpu'))

    print("Total loss for Model 1: {}".format(loss1))
    print("Average loss for Model 1: {}".format(av_loss1))
    print("Total loss for Model 2: {}".format(loss2))
    print("Total loss for Model 2: {}".format(av_loss2))


if __name__ == '__main__':
    compare_models("../../mse_results/train_len10000_mse_start0_seq10.pt",
                   "../../weighted_mse_results/weighted_train_len1000_mse_start0_seq10.pt")

