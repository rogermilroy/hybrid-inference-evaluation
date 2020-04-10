import time

import torch
from src.torchscript.hybrid_inference_models import ExtendedKalmanHybridInference

if __name__ == '__main__':

    H = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    Q = (1e4 * torch.tensor(
        [3.04167e-6, 3.04167e-6, 3.04167e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 3.04167e-8, 3.04167e-8,
         3.04167e-8, 1e-6, 1e-6, 1e-6])) * torch.eye(15)
    R = torch.tensor(
        [1e-6, 1e-6, 1.0, 1e-6, 1e-6, 1e-6, 100.0, 100.0, 1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6, 1e-6]) * torch.eye(15)

    ekhi = ExtendedKalmanHybridInference(H, Q, R)
    # need this because of batch norm...
    ekhi.eval()

    scripted_ekhi = torch.jit.script(ekhi)

    Fs = torch.stack([torch.eye(15)] * 100)
    tim = time.time()
    for i in range(1):
        print(i)

        obs = torch.zeros((1, 100, 15))
        # print(obs.shape)
        # print(Fs.shape)

        res = scripted_ekhi(obs, Fs)
        print(res)

    print("Time taken for 1: ", time.time() - tim)
    #
    scripted_ekhi.save("ekhi_model_larger.pt")

    # print(scripted_ekhi)
    # print(scripted_ekhi.predict)
    # print(scripted_ekhi.forward)
