from tqdm import tqdm
import torch
from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataloader import get_dataloaders
from torch.nn.functional import mse_loss
from src.models.graphical_model import KalmanGraphicalModel, KalmanInputGraphicalModel


def evaluate_model_input(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]]).t()
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states, inputs) in enumerate(loader):
            obs, states, inputs = obs.to(device), states.to(device), inputs.to(device)

            # compute the prediction.
            if isinstance(model, HybridInference):
                out, out_list = model(obs, inputs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), inputs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(0, 2, 1), states)

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = len(loader.dataset)
    return epoch_loss, (epoch_loss / divisor)


def evaluate_model(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]]).t()
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states) in enumerate(loader):
            obs, states = obs.to(device), states.to(device)

            # compute the prediction.
            if isinstance(model, HybridInference):
                out, out_list = model(obs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), 1e-4, 200)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(0, 2, 1), states)

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def compare_models(model_1, model_1_input, model_2, model_2_input, path_to_model1: str,
                   path_to_model2: str):
    # need datasets
    _, _, test_loader = get_dataloaders(train_samples=1000,
                                        val_samples=1000,
                                        test_samples=50,
                                        sample_length=10,
                                        starting_point=1,
                                        batch_size=5,
                                        extras=False)
    if model_2_input or model_1_input:
        _, _, test_loader_input = get_dataloaders(train_samples=1000,
                                                  val_samples=1000,
                                                  test_samples=50,
                                                  sample_length=10,
                                                  starting_point=1,
                                                  batch_size=5,
                                                  inputs=True,
                                                  extras=False)
    else:
        print("No inputs")
        test_loader_input = test_loader
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
    G = torch.tensor([[1 / 2, 1, 0., 0.],
                      [0., 0., 1 / 2, 1]]).t()
    if model_1_input:
        model1 = model_1(F=F,
                         H=H,
                         Q=Q,
                         R=R,
                         G=G)
    else:
        model1 = model_1(F=F,
                         H=H,
                         Q=Q,
                         R=R)
    if isinstance(model_1, HybridInference):
        model_1.gamma = 1e-4
        model1.load_state_dict(torch.load(path_to_model1, map_location=torch.device("cpu")))
    if model_2_input:
        model2 = model_2(F=F,
                         H=H,
                         Q=Q,
                         R=R,
                         G=G)
    else:
        model2 = model_2(F=F,
                         H=H,
                         Q=Q,
                         R=R)

    if isinstance(model_2, HybridInference):
        model_1.gamma = 1e-4
        model2.load_state_dict(torch.load(path_to_model2, map_location=torch.device("cpu")))

    # evaluate and print/return results.
    if model_1_input:
        loss1, av_loss1 = evaluate_model_input(model=model1, loader=test_loader_input,
                                               criterion=mse_loss,
                                               device=torch.device('cpu'))
    else:
        loss1, av_loss1 = evaluate_model(model=model1, loader=test_loader, criterion=mse_loss,
                                         device=torch.device('cpu'))
    if model_2_input:
        loss2, av_loss2 = evaluate_model_input(model=model2, loader=test_loader_input,
                                               criterion=mse_loss,
                                               device=torch.device('cpu'))
    else:
        loss2, av_loss2 = evaluate_model(model=model2, loader=test_loader, criterion=mse_loss,
                                         device=torch.device('cpu'))

    print("Total loss for Model 1: {}".format(loss1))
    print("Average loss for Model 1: {}".format(av_loss1))
    print("Total loss for Model 2: {}".format(loss2))
    print("Total loss for Model 2: {}".format(av_loss2))


if __name__ == '__main__':
    # compare_models(model_1=HybridInference, model_2=HybridInference, model_1_input=False,
    #                model_2_input=False,
    #                path_to_model1="../../mse_results/train_len10000_mse_start0_seq10.pt",
    #                path_to_model2="../../weighted_mse_results/weighted_train_len1000_mse_start0_seq10.pt")
    compare_models(model_1=KalmanInputGraphicalModel, model_2=HybridInference, model_1_input=True,
                   model_2_input=True,
                   path_to_model1="",
                   path_to_model2="../../weighted_mse_results"
                                  "/weighted_input_train_len5000_mse_start0_seq10.pt")
