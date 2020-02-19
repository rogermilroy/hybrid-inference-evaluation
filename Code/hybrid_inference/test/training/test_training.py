from unittest import TestCase
import torch
from src.training.train_hybrid_inference import train_hybrid_inference
from src.models.hybrid_inference_model import HybridInference
from src.data.synthetic_position_dataset import SyntheticPositionDataset
from torch.nn.functional import mse_loss
from src.training.loss import weighted_mse_loss
from time import time


class TestTraining(TestCase):

    def setUp(self) -> None:
        # Check if your system supports CUDA
        use_cuda = torch.cuda.is_available()

        # Setup GPU optimization if CUDA is supported
        if use_cuda:
            self.computing_device = torch.device("cuda")
            extras = {"num_workers": 7, "pin_memory": True}
            print("Using CUDA")
        else:  # Otherwise, train on the CPU
            self.computing_device = torch.device("cpu")
            extras = False
            print("CUDA NOT supported")
        self.H = torch.tensor([[1., 0., 0., 0.],
                               [0., 0., 1., 0.]]).to(self.computing_device)
        self.dataset = SyntheticPositionDataset(x0=torch.tensor([0., 0.1, 0., 0.1]), n_samples=100, sample_length=10, starting_point=0, seed=42)
        self.data_params = dict()
        self.data_params["train_samples"] = 10
        self.data_params["val_samples"] = 20
        self.data_params["test_samples"] = 50
        self.data_params["sample_length"] = 100
        self.data_params["starting_point"] = 1000
        self.data_params["batch_size"] = 4
        self.data_params["extras"] = extras


    def test_training_size_len_weighted(self):
        """
        This will train 3 models of varying training sample amounts. starts at 1000 ends at 100000
        Each will output a file of results.
        :return:
        """
        for i in range(1):
            path = "./fixed_len" + str(self.data_params["train_samples"]) + \
                   "start1000_seq100"
            train_hybrid_inference(epochs=20, val=True, loss=weighted_mse_loss, weighted=True,
                                   log_path=path + ".txt",
                                   save_path=path + ".pt",
                                   data_params=self.data_params,
                                   computing_device=self.computing_device,
                                   inputs=False)
            self.data_params["train_samples"] *= 10

    # def test_sample_seq_len_weighted(self):
    #     """
    #     This will train 3 models of varying sequence lengths. starts at 10 ends at 1000
    #     Each will output a file of results.
    #     :return:
    #     """
    #     for i in range(3):
    #         path = "./weighted_train_len1000_mse_start0_seq" + str(self.data_params[
    #                                                                    "sample_length"])
    #         train_hybrid_inference(epochs=50, val=True, loss=weighted_mse_loss, weighted=True,
    #                                log_path=path + ".txt",
    #                                save_path=path + ".pt",
    #                                data_params=self.data_params,
    #                                computing_device=self.computing_device,
    #                                inputs=False)
    #         self.data_params["sample_length"] *= 10
    #
    # def test_sample_start_weighted(self):
    #     """
    #     This will train 3 models of varying sequence start points. starts at 1 ends at 1000
    #     Each will output a file of results.
    #     :return:
    #     """
    #     for i in range(4):
    #         path = "./weighted_train_len1000_mse_start" + str(self.data_params["starting_point"])\
    #                + \
    #                "_seq10"
    #         train_hybrid_inference(epochs=50, val=True, loss=weighted_mse_loss, weighted=True,
    #                                log_path=path + ".txt",
    #                                save_path=path + ".pt",
    #                                data_params=self.data_params,
    #                                computing_device=self.computing_device,
    #                                inputs=False)
    #         self.data_params["train_samples"] *= 10
    #
    # def test_training_size_len(self):
    #     """
    #     This will train 3 models of varying training sample amounts. starts at 1000 ends at 100000
    #     Each will output a file of results.
    #     :return:
    #     """
    #     for i in range(3):
    #         path = "./train_len" + str(self.data_params["train_samples"]) + \
    #                "_mse_start0_seq10"
    #         train_hybrid_inference(epochs=50, val=True, loss=mse_loss, weighted=False,
    #                                log_path=path + ".txt",
    #                                save_path=path + ".pt",
    #                                data_params=self.data_params,
    #                                computing_device=self.computing_device,
    #                                inputs=False)
    #         self.data_params["train_samples"] *= 10
    #
    # def test_sample_seq_len(self):
    #     """
    #     This will train 3 models of varying sequence lengths. starts at 10 ends at 1000
    #     Each will output a file of results.
    #     :return:
    #     """
    #     for i in range(3):
    #         path = "./train_len1000_mse_start0_seq" + str(self.data_params["sample_length"])
    #         train_hybrid_inference(epochs=50, val=True, loss=mse_loss, weighted=False,
    #                                log_path=path + ".txt",
    #                                save_path=path + ".pt",
    #                                data_params=self.data_params,
    #                                computing_device=self.computing_device,
    #                                inputs=False)
    #         self.data_params["sample_length"] *= 10
    #
    # def test_sample_start(self):
    #     """
    #     This will train 3 models of varying sequence start points. starts at 1 ends at 1000
    #     Each will output a file of results.
    #     :return:
    #     """
    #     for i in range(4):
    #         path = "./train_len1000_mse_start"+str(self.data_params["starting_point"])+"_seq10"
    #         train_hybrid_inference(epochs=50, val=True, loss=mse_loss, weighted=False,
    #                                log_path=path + ".txt",
    #                                save_path=path + ".pt",
    #                                data_params=self.data_params,
    #                                computing_device=self.computing_device,
    #                                inputs=False)
    #         self.data_params["train_samples"] *= 10

    # def test_trained_model(self):
    #     model = HybridInference(F=torch.tensor([[1., 1., 0., 0.],
    #                                             [0., 1., 0., 0.],
    #                                             [0., 0., 1., 1.],
    #                                             [0., 0., 0., 1.]]).to(self.computing_device),
    #                             H=torch.tensor([[1., 0., 0., 0.],
    #                                             [0., 0., 1., 0.]]).to(self.computing_device),
    #                             Q=torch.tensor([[0.05 ** 2, 0., 0., 0.],
    #                                             [0., 0.05 ** 2, 0., 0.],
    #                                             [0., 0., 0.05 ** 2, 0.],
    #                                             [0., 0., 0., 0.05 ** 2]]).to(self.computing_device),
    #                             R=(0.05 ** 2) * torch.eye(2).to(self.computing_device),
    #                             gamma=1e-4)
    #     model.load_state_dict(torch.load("./hybrid_inference_mse_params.pt"))
    #     model.to(self.computing_device)
    #     obs, states = self.dataset[4]
    #     obs, states = obs.to(self.computing_device), states.to(self.computing_device)
    #     best_estimate = model(obs, 100)
    #     print(states - best_estimate.t())

    # def test_continue_training(self):
    #     train_hybrid_inference(epochs=1, val=False, loss=mse_loss, weighted=False
    #                            save_path="./hybrid_inference_mse_params.pt",
    #                            load_model="./hybrid_inference_mse_params.pt")
    #
    # def test_weighted_loss(self):
    #     train_hybrid_inference(epochs=100, val=True, loss=weighted_mse_loss, weighted=False
    #                            save_path="./hybrid_inference_weighted_params.pt")
