import torch
import torch.nn as nn
from torch import tensor
from torch.nn import GRUCell

class BatchGRUCell(nn.Module):
    """
    This class implements batching over the top of the existing GRUCell.
    This is to allow batching of GNN sequences specifically.
    It splits any input into 2d tensors which should then be sequence x feature or batch x feature.
    if using it to batch batches.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialises the component GRUCell.
        :param input_size: int Dimension of the feature
        :param hidden_size: int Dimension of the hidden state, also output dim.
        :param bias: bool Whether the GRU should have bias terms or not.
        """
        super(BatchGRUCell, self).__init__()
        self.gru = GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

    def forward(self, x: tensor, hidden: tensor = None) -> tensor:
        """
        For each batch it passes it through the GRUCell then concatenates the results and returns them.
        :param x: Tensor The input tensor. Batch x sequence x feature dim
        :param x: Tensor The previous hidden state. Batch x sequence x hidden dim. Optional,
        defaults to 0.
        :return: Tensor The resulting hidden states. Batch x sequence x
        """
        if len(x.shape) == 3:
            result = list()
            if hidden is not None:
                for batch, h in zip(x, hidden):
                    result.append(self.gru(batch, h))
                result = torch.stack(result)
            else:
                for batch in x:
                    result.append(self.gru(batch))
                result = torch.stack(result)
        else:
            raise ValueError("This unit only accepts 3d Tensors.")
        return result
