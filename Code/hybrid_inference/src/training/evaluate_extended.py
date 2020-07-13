import torch
from src.data.gazebo_dataloader import get_dataloaders
from src.models.graphical_model import ExtendedKalmanGraphicalModel
from src.models.hybrid_inference_model import ExtendedKalmanHybridInference
from src.utils.data_converters import torch2numpy, torchseq2numpyseq, numpy2torch
from src.training.extended_kalman import QuadcopterExtendedKalman
from torch.nn.functional import mse_loss
from tqdm import tqdm
import numpy as np
from copy import deepcopy


def evaluate_extended_model(model, loader, criterion, device, Hs=None, vis_example=0):
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

    if Hs is None:
        Hs = torch.stack([torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
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
                      ], device=device)] * loader.batch_size)

    if vis_example > 0:
        sample = len(loader) / vis_example
    with torch.no_grad():
        for num, (ys, Fs, gts) in tqdm(enumerate(loader)):
            ys, Fs, gts = ys.to(device), Fs.squeeze(0).to(device), gts.to(device)

            # compute the prediction.
            out, out_list = model(ys, Fs, Hs)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", gts.permute(0, 2, 1))
                print("Difference: ", X @ gts.permute(0, 2, 1) - X @ out)

            loss = criterion(X @ out, X @ gts.permute(0, 2, 1), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def evaluate_extended_model_h(model, loader, criterion, device, vis_example=0):
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
        for num, (ys, Fs, Hs, gts) in tqdm(enumerate(loader)):
            ys, Fs, Hs, gts = ys.to(device), Fs.squeeze(0).to(device), Hs.squeeze(0).to(device), gts.to(device)

            # compute the prediction.
            out, out_list = model(ys, Fs, Hs)

            if sample and num % sample == 0:
                print("Predictions: ", out)
                print("Ground truth: ", gts.permute(0, 2, 1))
                print("Difference: ", X @ gts.permute(0, 2, 1) - X @ out)

            loss = criterion(X @ out, X @ gts.permute(0, 2, 1), reduction='sum')

            # add to the epochs loss
            epoch_loss += float(loss)

    divisor = loader.dataset.total_samples()
    return epoch_loss, (epoch_loss / divisor)


def H(x):
    return np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
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
                      ])


def Hx(x):
    return H(x).dot(x)


def run_kalman(filter: QuadcopterExtendedKalman, ys):
    xs = list()
    for y in ys:
        filter.predict()
        filter.update(y, HJacobian=H, Hx=Hx)  # both H because using linear H.
        xs.append(deepcopy(filter.x))
    return np.stack(xs, axis=0)


def compare2kalman(model, loader):
    """
    Compare HI to classical implementation of Extended Kalman Filter
    :return:
    """
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
                      ])

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
                  ])

    kal_tot_loss = 0.
    hi_tot_loss = 0.
    for num, (ys, Fs, gts) in tqdm(enumerate(loader)):  # TODO add Hs if necessary
        Hs = torch.stack([H] * ys.shape[1])
        # set up Kalman Filter things and compute Kalman loss
        filter = QuadcopterExtendedKalman(dim_x=15, dim_y=11, Fs=torchseq2numpyseq(Fs.squeeze(0)))
        x_0 = gts[0][0]  # TODO check
        filter.x = torch2numpy(x_0)

        # carry out the filter over the sequences and Fs
        kal_states = run_kalman(filter, torchseq2numpyseq(ys))

        kal_loss = mse_loss(X @ numpy2torch(kal_states).unsqueeze(0).permute(0,2,1),
                            X @ gts.permute(0,2,1),
                            reduction='sum')

        kal_tot_loss += float(kal_loss)

        # compute HI loss
        hi_state, _ = model( ys.permute(0, 1, 2), Fs.squeeze(0), Hs)
        hi_loss = mse_loss(X @ hi_state, X @ gts.permute(0,2,1), reduction='sum')

        hi_tot_loss += float(hi_loss)

    # print results
    divisor = loader.dataset.total_samples()
    print("Total loss for HI Model: {}".format(hi_tot_loss))
    print("Average loss for HI Model: {}".format(hi_tot_loss / divisor))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss))
    print("Total loss for Kalman Model: {}".format(kal_tot_loss / divisor))


if __name__ == '__main__':
    # H = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.]) * torch.eye(15)
    # Q = (1e1 * torch.tensor(
    #     [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18,
    #      0.18])) * torch.eye(15)
    # R = torch.tensor(
    #     [1.0, 1.0, 10.0, 100.0, 100.0, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) * torch.eye(11)


    # Q = (1e1 * torch.tensor(
    #     [1., 1., 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18, 0.18],
    #     device=torch.device("cpu"))) * torch.eye(15, device=torch.device("cpu"))
    # R = torch.tensor(
    #     [10.0, 10.0, 100.0, 100.0, 100.0, 1e-2, 1e-2, 1e-2], device=torch.device("cpu")) * torch.eye(8,
    #                                                                                                  device=torch.device(
    #                                                                                                      "cpu"))

    # ekhi = ExtendedKalmanHybridInference(Q, R, gamma=2e-4)
    # train, val, test = get_dataloaders("../../../catkin_ws/recording2/", 5000, 0.1, 0.01, 100,
    #                                    seed=12, H=False)

    # for num, (ys, Fs, gts) in enumerate(test):
    #     if num == 1:
    #         break
    #     print(gts[0, 90])
    #     print(Fs[0, 90] @ gts[0, 90])
    #     print(Fs[0, 90] @ (H.t() @ ys[0, 90]))
    # print((X @ gts.permute(0, 2, 1)).size())
    # F = torch.jit.load('../../../catkin_ws/recording1/F-120.pt').named_parameters()
    # next(F)
    # Fm = next(F)[1]
    #
    # x = torch.tensor([0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    # print(x)
    # print(Fm)
    # print(Fm @ x)
    # print(Fm @ Fm @ x)
    # print(Fm @ Fm @ Fm @ x)

    print(torch.load('../../../catkin_ws/recording_long/odom-4.pt'))

    y = torch.jit.load('../../../catkin_ws/recording_long/y-3.pt').named_parameters()
    print(next(y)[1])
    print(next(y)[1])

    F = torch.jit.load('../../../catkin_ws/recording_long/F-3.pt').named_parameters()
    print(next(F)[1])
    print(next(F)[1])

    # alpha, beta, gamma, x, y, z, x_vel, y_vel, z_vel, alpha_rate, beta_rate, gamma_rate, x_accel, y_accel,\
    # z_accel = sympy.symbols('alpha, beta, gamma, x, y, z, x_vel, y_vel, z_vel, alpha_rate, beta_rate, gamma_rate, x_accel, y_accel, z_accel')
    #
    # H = sympy.Matrix([x,
    #                   y,
    #                   x_vel,
    #                   y_vel,
    #                   sympy.sqrt(x_accel ** 2 + y_accel**2 + z_accel**2) * sympy.sin((alpha)),
    #                   sympy.sqrt(x_accel ** 2 + y_accel**2 + z_accel**2) * sympy.sin((beta)),
    #                   sympy.sqrt(((sympy.sqrt(x_accel ** 2 + y_accel**2 + z_accel**2) * sympy.cos(alpha))**2) - (sympy.sqrt(x_accel ** 2 + y_accel**2 + z_accel**2) * sympy.sin(beta))**2)])
    #
    # x = sympy.Matrix([alpha, beta, gamma, x, y, z, x_vel, y_vel, z_vel, alpha_rate, beta_rate, gamma_rate, x_accel, y_accel, z_accel])
    #
    # J = H.jacobian(x)
    # print(H)
    # print(J)

    # [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [sqrt(x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_accel*sin(alpha)/sqrt(x_accel**2 + y_accel**2 + z_accel**2), y_accel*sin(alpha)/sqrt(x_accel**2 + y_accel**2 + z_accel**2), z_accel*sin(alpha)/sqrt(x_accel**2 + y_accel**2 + z_accel**2)],
    # [0, sqrt(x_accel**2 + y_accel**2 + z_accel**2)*cos(beta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_accel*sin(beta)/sqrt(x_accel**2 + y_accel**2 + z_accel**2), y_accel*sin(beta)/sqrt(x_accel**2 + y_accel**2 + z_accel**2), z_accel*sin(beta)/sqrt(x_accel**2 + y_accel**2 + z_accel**2)],
    # [-(x_accel**2 + y_accel**2 + z_accel**2)*sin(alpha)*cos(alpha)/sqrt(-(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)**2 + (x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha)**2), -(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)*cos(beta)/sqrt(-(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)**2 + (x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha)**2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (-x_accel*sin(beta)**2 + x_accel*cos(alpha)**2)/sqrt(-(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)**2 + (x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha)**2), (-y_accel*sin(beta)**2 + y_accel*cos(alpha)**2)/sqrt(-(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)**2 + (x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha)**2), (-z_accel*sin(beta)**2 + z_accel*cos(alpha)**2)/sqrt(-(x_accel**2 + y_accel**2 + z_accel**2)*sin(beta)**2 + (x_accel**2 + y_accel**2 + z_accel**2)*cos(alpha)**2)]]

    # print(evaluate_extended_model(ekhi, test, mse_loss, "cpu"))
    # compare2kalman(ekhi, test)
