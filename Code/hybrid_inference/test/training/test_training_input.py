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
        self.data_params["train_samples"] = 1000
        self.data_params["val_samples"] = 200
        self.data_params["test_samples"] = 500
        self.data_params["sample_length"] = 10
        self.data_params["starting_point"] = 1
        self.data_params["batch_size"] = 5
        self.data_params["extras"] = extras

    def test_training_size_len_weighted(self):
        """
        This will train 3 models of varying training sample amounts. starts at 1000 ends at 100000
        Each will output a file of results.
        :return:
        """
        for i in range(1):
            path = "./weighted_train_len" + str(self.data_params["train_samples"]) + \
                   "_mse_start0_seq10"
            train_hybrid_inference(epochs=50, val=True, loss=weighted_mse_loss, weighted=True,
                                   log_path=path + ".txt",
                                   save_path=path + ".pt",
                                   data_params=self.data_params,
                                   computing_device=self.computing_device,
                                   inputs=True)
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
