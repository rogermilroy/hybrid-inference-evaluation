import math
from time import time

import torch
from src.data.gazebo_dataloader import get_dataloaders
from src.models.hybrid_inference_model import ExtendedKalmanHybridInference
from src.training.evaluate_extended import evaluate_extended_model
from src.training.loss import weighted_mse_loss
from torch.nn.functional import mse_loss
from torch.optim import Adam


def train_model(model, loader, optimizer, device, weighted, validate_every, n,
                val_loader, log_file, save_path):
    """
    Method to train for one epoch.
    """
    model.train()
    epoch_loss = 0.
    losses = list()
    best_model = None
    best_val = math.inf

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

    for num, (ys, Fs, gts) in enumerate(loader):
        ys, Fs, gts = ys.to(device), Fs.squeeze(0).to(device), gts.to(device)

        # zero the optimizers gradients from the previous iteration.
        optimizer.zero_grad()

        # compute the prediction.
        out, out_list = model(ys, Fs)
        t = list()
        for o in out_list:
            t.append(X @ o)

        # compute the loss
        if weighted:
            loss = weighted_mse_loss(torch.stack(t), (X @ gts.permute(0, 2, 1)).permute(0, 2, 1))
        else:
            loss = mse_loss(X @ out, X @ gts.permute(0, 2, 1))

        # propagate the loss back through the network.
        loss.backward()

        # update the weights.
        optimizer.step()

        # add to the epochs loss
        losses.append(float(loss))
        epoch_loss += float(loss)

        # in epoch validation
        if num + 1 % validate_every == 0 and num != 0:
            # if we are validating then do that and print the results
            _, val_av_loss = evaluate_extended_model(model=model, loader=val_loader,
                                                     criterion=mse_loss,
                                                     device=device)
            print("Batch {} avg validation loss: {}".format(num + 1, val_av_loss))
            log_file.write("After {} : {}\n".format(num + 1, val_av_loss))

            # check better only save better ones.
            # store the best model based on validation scores.
            if val_av_loss < best_val:
                best_val = val_av_loss
                best_model = model.state_dict()
                # save the model at this point.
                torch.save(model.state_dict(), save_path)
            if num + 1 == n:
                break
        if num + 1 == n:
            break

    return epoch_loss, losses, best_model


def train_hybrid_inference(n, weighted, save_path, log_path="./training.txt", vis_examples=0,
                           load_model=None, computing_device=torch.device("cpu")):
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

    model = ExtendedKalmanHybridInference(H, Q, R, gamma=2e-4)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))

    model.to(computing_device)

    print("Model on CUDA?", next(model.parameters()).is_cuda)

    optimizer = Adam(model.parameters(), lr=0.01)  # reduce the learning rate to 0.0001 from 0.001

    train_loader, val_loader, test_loader = get_dataloaders("../../../catkin_ws/recording2/",
                                                            dataset_len=5000, val=0.02, test=0.02, sample_len=50,
                                                            batch_size=1)

    divisor = train_loader.dataset.total_samples()

    _, pre_val_av = evaluate_extended_model(model, test_loader, criterion=mse_loss, device=computing_device)

    print("Pre validation average loss: {}".format(pre_val_av))

    validate_every = 100
    with open(log_path, 'w+') as log_file:

        # print("Before training avg test loss: {}".format(pre_val / divisor))
        # log_file.write("Before training avg test loss: {}".format(pre_val / divisor))

        start = time()
        # train a simple epoch and record and print the losses.

        epoch_loss, epoch_losses, best_model = train_model(model=model, loader=train_loader,
                                                           optimizer=optimizer,
                                                           device=computing_device,
                                                           weighted=weighted,
                                                           validate_every=validate_every,
                                                           n=n,
                                                           val_loader=val_loader,
                                                           log_file=log_file, save_path=save_path)
        print("Losses = ", epoch_losses)

        if best_model is not None:
            model.load_state_dict(best_model)

        # test it.

        test_loss, test_av_loss = evaluate_extended_model(model=model,
                                                          loader=test_loader,
                                                          criterion=mse_loss,
                                                          device=computing_device,
                                                          vis_example=vis_examples)

        print("Time taken: {}".format(time() - start))
        print("Test average loss: {}".format(test_av_loss))
        # log_file.write("Time taken: {}\n".format(time() - start))
        log_file.write("Test : {}\n".format(test_av_loss))


if __name__ == '__main__':
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Trying CUDA")

    train_hybrid_inference(500, weighted=True, save_path="../torchscript/ekhi_trained_1000.pt", computing_device=device)
