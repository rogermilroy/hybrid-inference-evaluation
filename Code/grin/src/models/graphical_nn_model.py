from torch import nn
import torch
from .encoding_model import GraphEncoderMLP
from .decoding_model import GraphDecoder

########################################################################################################################
# This code was written with reference to vgsatorras hybrid inference code.
# https://github.com/vgsatorras/hybrid-inference
#
# Almost everything is different but it was invaluable for understanding how to implement the paper:
# Combining Generative and Discriminative Models for Hybrid Inference by Sartorras, Akata and Welling. 20 Jun 2019
#
########################################################################################################################


class KalmanGNN(nn.Module):

    def __init__(self, h_dim, x_dim, y_dim):
        super(KalmanGNN, self).__init__()
        # initialise hs? outside of the GNN and just have GNN process?
        # provide a tool for initialising hs
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        # convolution to transform observations into hys
        self.hy_initialiser = torch.nn.Conv1d(y_dim * 2, h_dim, kernel_size=1)
        self.hy = None

        # mlps for edge encoding.
        self.past_curr_nn = GraphEncoderMLP(2 * self.h_dim + self.x_dim, self.h_dim, self.h_dim)
        self.future_curr_nn = GraphEncoderMLP(2 * self.h_dim + self.x_dim, self.h_dim, self.h_dim)
        self.y_curr_nn = GraphEncoderMLP(2 * self.h_dim + self.x_dim, self.h_dim, self.h_dim)

        # mlp for node encoding. TODO do I need three layers for this one?
        self.node_nn = GraphEncoderMLP(self.h_dim, self.h_dim, self.h_dim)

        # gru for recurrence
        self.gru = nn.GRUCell(input_size=self.h_dim, hidden_size=self.h_dim)

        self.decoder = GraphDecoder(self.h_dim, self.h_dim, self.x_dim)

    def forward(self, hx, graphical_messages):
        """
        Forward pass over the GNN.
        :param hx:
        :param graphical_messages:
        :return:
        """
        past_curr_mess, fut_curr_mess, y_curr_mess = graphical_messages
        # check dims and transpose if necessary TODO change to allow batches. also many other places.
        if hx.shape[0] == self.h_dim:
            hx = hx.t()

        # construct the edges to pass through the relevant models
        hx_past = torch.cat([hx[0].unsqueeze(0), hx[:-1]]).t()
        hx_future = torch.cat([hx[1:], hx[-1].unsqueeze(0)]).t()
        hx = hx.t()

        # pass compute edge encodings.
        past_curr_edge = self.past_curr_nn(torch.cat([hx_past, hx, past_curr_mess]).t()).t()
        fut_curr_edge = self.future_curr_nn(torch.cat([hx_future, hx, fut_curr_mess]).t()).t()
        y_curr_edge = self.y_curr_nn(torch.cat([self.hy, hx, y_curr_mess]).t()).t()

        # sum edge encodings?
        # pass through node encoder
        h = self.node_nn(sum([past_curr_edge, fut_curr_edge, y_curr_edge]).t()).t()

        # pass through gru
        h = self.gru(h.t(), hx.t()).t()

        # return the decoded hx and hx.
        eps = self.decoder(h.unsqueeze(0)).squeeze(0)

        return eps, h

    def initialise_hx_y(self, ys):
        """
        A utility to be accessed by the containing class.
        Initialises the hx and returns them.
        Initialises the hys and stores them.?
        :param ys: observations?
        :return:
        """
        if ys.shape[0] == self.y_dim:
            # turn from dims x samples to samples x dims
            ys = ys.t()
        num_samples = ys.shape[0]
        # take the difference between past and current ys and current and future ys.
        y_past = torch.cat([ys[0].unsqueeze(0), ys[:-1]]).t()
        y_future = torch.cat([ys[1:], ys[-1].unsqueeze(0)]).t()
        ys = ys.t()
        diff1 = ys - y_past
        diff2 = y_future - ys

        # concatenate and  unsqueeze
        hy_in = torch.cat([diff1, diff2]).unsqueeze(0)
        # pass through a one dimensional convolution.
        # save as self.hy
        self.hy = self.hy_initialiser(hy_in).squeeze(0)  # needs squeezing as convs only take 3d inputs
        hx = torch.randn((self.h_dim, num_samples))

        return hx
