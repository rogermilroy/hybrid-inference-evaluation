from src.data.gazebo_dataloader import get_dataloaders
from src.models.hybrid_inference_model import ExtendedKalmanHybridInference
from src.utils.data_converters import *
from tqdm import tqdm


def evaluate_extended_model(model, loader, criterion, device, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None

    X = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
                      ], device=device)

    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (ys, Fs, gts) in tqdm(enumerate(loader)):
            ys, Fs, gts = ys.to(device), Fs.squeeze(0).to(device), gts.to(device)

            # compute the prediction.
            out, out_list = model(ys, Fs)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", gts.permute(0, 2, 1))
                print("Difference: ", X @ gts.permute(0, 2, 1) - X @ out)

            loss = criterion(X @ out, X @ gts.permute(0, 2, 1), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


if __name__ == '__main__':
    # H = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.]) * torch.eye(15)
    # Q = (1e1 * torch.tensor(
    #     [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18,
    #      0.18])) * torch.eye(15)
    # R = torch.tensor(
    #     [1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]) * torch.eye(15)

    H = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                      ], device=torch.device("cpu"))
    Q = (1e1 * torch.tensor(
        [1., 1., 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18, 0.18],
        device=torch.device("cpu"))) * torch.eye(15, device=torch.device("cpu"))
    R = torch.tensor(
        [10.0, 10.0, 100.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], device=torch.device("cpu")) * torch.eye(11,
                                                                                                                   device=torch.device(
                                                                                                                       "cpu"))

    ekhi = ExtendedKalmanHybridInference(H, Q, R, gamma=2e-4)
    train, val, test = get_dataloaders("../../../catkin_ws/recording2/", 5000, 0.1, 0.01, 100, seed=12)

    for num, (ys, Fs, gts) in enumerate(test):
        if num == 1:
            break
        print(gts[0, 90])
        print(Fs[0, 90] @ gts[0, 90])
        print(Fs[0, 90] @ (H.t() @ ys[0, 90]))
        # print((X @ gts.permute(0, 2, 1)).size())

    # print(evaluate_extended_model(ekhi, test, mse_loss, "cpu"))
