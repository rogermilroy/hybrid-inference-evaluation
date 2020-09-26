import time

import torch
from src.torchscript.hybrid_inference_models import ExtendedKalmanHybridInference

if __name__ == '__main__':

    # H = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    #                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #                   ], device=torch.device("cpu"))
    Q = (1e1 * torch.tensor(
        [1., 1., 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18, 0.18],
        device=torch.device("cpu"))) * torch.eye(15, device=torch.device("cpu"))
    R = torch.tensor(
        [10.0, 10.0, 100.0, 1.0, 1.0, 1e-2, 1e-2, 1e-2], device=torch.device("cpu")) * torch.eye(8, device=torch.device(
        "cpu"))

    # H = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.]) * torch.eye(15)
    #
    # Q = (1e1 * torch.tensor(
    #     [1., 1., 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.00548311, 0.00548311, 0.00548311, 0.18, 0.18, 0.18])) * torch.eye(15)
    # R = torch.tensor(
    #     [1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1e-4, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) * torch.eye(15)

    ekhi = ExtendedKalmanHybridInference(Q, R, gamma=2e-4)
    # need this because of batch norm...
    ekhi.eval()

    scripted_ekhi = torch.jit.script(ekhi)

    Fs = torch.stack([torch.eye(15)] * 100)
    Hs = torch.stack([torch.zeros((8, 15))] * 100)
    tim = time.time()
    for i in range(1):
        print(i)

        obs = torch.zeros((1, 100, 8))
        # print(obs.shape)
        # print(Fs.shape)

        res = scripted_ekhi(obs, Fs, Hs)
        print(res)

    print("Time taken for 1: ", time.time() - tim)

    scripted_ekhi.save("ekhi_model_redone.pt")

    # print(scripted_ekhi)
    # print(scripted_ekhi.predict)
    # print(scripted_ekhi.forward)
