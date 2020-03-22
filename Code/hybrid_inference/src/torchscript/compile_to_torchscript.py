import torch
from src.models.hybrid_inference_model import HybridInference

if __name__ == '__main__':
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

    hi = HybridInference(F, H, Q, R)

    scripted_hi = torch.jit.script(hi)

    scripted_hi.save("test_model.pt")

    print(scripted_hi.forward.code)
