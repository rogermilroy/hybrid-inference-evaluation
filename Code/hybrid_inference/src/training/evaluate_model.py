from src.data.synthetic_position_dataloader import get_dataloaders
from src.models.graphical_model import KalmanGraphicalModel
from src.models.hybrid_inference_model import KalmanHybridInference
from src.utils.data_converters import *
from torch.nn.functional import mse_loss


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
            if isinstance(model, KalmanHybridInference):
                out, out_list = model(obs, inputs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), inputs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            loss = criterion(out.permute(1, 2, 0), states.permute(2, 1, 0), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
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
            if isinstance(model, KalmanHybridInference):
                out, out_list = model(obs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            # need to override mse averaging.
            loss = criterion(out.permute(1, 2, 0), states.permute(2, 1, 0), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def evaluate_model_predict(model, loader, criterion, device, n, vis_example=0):
    model.eval()
    epoch_loss = 0.
    sample = None
    H = torch.tensor([[1., 0., 0., 0.],
                      [0., 0., 1., 0.]])
    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (obs, states) in enumerate(loader):
            obs, states = obs.to(device), states.to(device)

            # modify obs to change the last n to 0s
            obs = obs[:, :-n, :]

            # compute the prediction.
            if isinstance(model, KalmanHybridInference):
                out, out_list = model.predict(n, obs)
            else:
                xs = H.matmul(obs.permute(0, 2, 1))
                out = model.iterate(xs, obs.permute(0, 2, 1), 1e-4)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", states)
                print("Difference: ", states - out)

            # need to override mse averaging.
            loss = criterion(torch.matmul(H, out.permute(0, 1, 2)), torch.matmul(H, states.permute(0, 2, 1)),
                             reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def compare_models(model_1, model_1_input, model_2, model_2_input, path_to_model1: str,
                   path_to_model2: str):
    # need datasets
    _, _, test_loader = get_dataloaders(train_samples=10,
                                        val_samples=10,
                                        test_samples=20,
                                        sample_length=100,
                                        starting_point=1000,
                                        batch_size=5,
                                        extras=False)
    if model_2_input or model_1_input:
        _, _, test_loader_input = get_dataloaders(train_samples=10,
                                                  val_samples=10,
                                                  test_samples=20,
                                                  sample_length=100,
                                                  starting_point=1000,
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
    if isinstance(model1, KalmanHybridInference):
        model1.gamma = 1e-4
        if len(path_to_model1) > 0:
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

    if isinstance(model2, KalmanHybridInference):
        model2.gamma = 1e-4

        if len(path_to_model2) > 0:
            print("pretrained")
            model2.load_state_dict(torch.load(path_to_model2, map_location=torch.device("cpu")))
        else:
            print("not pretrained")

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


def compare2kalman(model, filter, loader):
    """
    Compare HI to classical implementation of Kalman Filter
    :return:
    """
    kal_tot_loss = 0.
    hi_tot_loss = 0.
    for num, (obs, states) in enumerate(loader):
        # set up Kalman Filter things and compute Kalman loss
        x_0 = states[0][0]  # TODO check
        filter.x = torch2numpy(x_0)

        mu, cov, _, _ = filter.batch_filter(torchseq2numpyseq(obs))
        kal_state, P, _, _ = filter.rts_smoother(mu, cov)
        kal_loss = mse_loss(numpy2torch(kal_state).unsqueeze(0), states, reduction='sum')

        kal_tot_loss += float(kal_loss)

        # compute HI loss
        hi_state, _ = model(obs)
        hi_loss = mse_loss(hi_state, states.permute(0, 2, 1), reduction='sum')

        hi_tot_loss += float(hi_loss)

    # print results
    divisor = loader.dataset.total_samples()
    print("Total loss for HI Model: {}".format(hi_tot_loss))
    print("Average loss for HI Model: {}".format(hi_tot_loss / divisor))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss / divisor))


if __name__ == '__main__':
    compare_models(model_1=KalmanGraphicalModel, model_2=KalmanHybridInference, model_1_input=True,
                   model_2_input=False,
                   path_to_model1="",
                   path_to_model2="")

    # compare_models(model_1=KalmanInputGraphicalModel, model_2=KalmanHybridInference, model_1_input=True,
    #                model_2_input=True,
    #                path_to_model1="",
    #                path_to_model2="../../weighted_mse_results"
    #                               "/weighted_input_train_len5000_mse_start0_seq10.pt")

    # F = torch.tensor([[1., 1., 0., 0.],
    #                   [0., 1., 0., 0.],
    #                   [0., 0., 1., 1.],
    #                   [0., 0., 0., 1.]])
    # H = torch.tensor([[1., 0., 0., 0.],
    #                   [0., 0., 1., 0.]])
    # Q = torch.tensor([[0.05 ** 2, 0., 0., 0.],
    #                   [0., 0.05 ** 2, 0., 0.],
    #                   [0., 0., 0.05 ** 2, 0.],
    #                   [0., 0., 0., 0.05 ** 2]])
    # R = (0.05 ** 2) * torch.eye(2)
    # G = torch.tensor([[1 / 2, 1, 0., 0.],
    #                   [0., 0., 1 / 2, 1]]).t()
    # P = np.eye(4) * 1000
    #
    # hi = KalmanHybridInference(F, H, Q, R)

    # kal = KalmanFilter(dim_x=4, dim_z=2)
    # kal.F = torch2numpy(F)
    # kal.H = torch2numpy(H)
    # kal.Q = torch2numpy(Q)
    # kal.R = torch2numpy(R)
    # kal.P = P
    #
    # _, _, test_loader = get_dataloaders(train_samples=10,
    #                                     val_samples=10,
    #                                     test_samples=200,
    #                                     sample_length=100,
    #                                     starting_point=1000,
    #                                     batch_size=1,
    #                                     extras=False)
    #
    # evaluate_model_predict(hi, test_loader, mse_loss, 'cpu', 2)

    # compare2kalman(hi, kal, test_loader)
