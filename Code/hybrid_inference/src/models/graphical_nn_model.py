from torch import nn
import torch
from .encoding_model import GraphEncoderMLP
from .decoding_model import GraphDecoder
from .batch_gru import BatchGRUCell

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
        self.gru = BatchGRUCell(input_size=self.h_dim, hidden_size=self.h_dim)

        self.decoder = GraphDecoder(self.h_dim, self.h_dim, self.x_dim)

    def forward(self, hx, graphical_messages):
        """
        Forward pass over the GNN.
        :param hx:
        :param graphical_messages:
        :return:
        """
        # messages batch x feat x samples
        past_curr_mess, fut_curr_mess, y_curr_mess = graphical_messages
        if len(hx.shape) == 3:
            # TODO work out dimension checking, for now just do it right.
            # from batch x feat x samples to samples x batch x feat
            hx = hx.permute(2, 0, 1)

            # create the time shifted xs. ensure that all are oriented batch x samples x feat now.
            hx_past = torch.cat([hx[0].unsqueeze(0), hx[:-1]]).permute(1, 2, 0)
            hx_future = torch.cat([hx[1:], hx[-1].unsqueeze(0)]).permute(1, 2, 0)
            hx = hx.permute(1, 2, 0)

            past_curr_edge = self.past_curr_nn(
                                    torch.cat([hx_past, hx, past_curr_mess], dim=1).permute(0, 2, 1))

            fut_curr_edge = self.future_curr_nn(
                                    torch.cat([hx_future, hx, fut_curr_mess], dim=1).permute(0, 2, 1))

            y_curr_edge = self.y_curr_nn(
                                torch.cat([self.hy, hx, y_curr_mess], dim=1).permute(0, 2, 1))
            # still batch x samples x features

            # sum edge encodings
            # pass through node encoder
            U = self.node_nn(
                        sum([past_curr_edge, fut_curr_edge, y_curr_edge]))

            # pass through gru
            h = self.gru(U, hx.permute(0, 2, 1))
            h = h.permute(0, 2, 1)

            eps = self.decoder(h)
        else:
            # check dims and transpose if necessary
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
            U = self.node_nn(sum([past_curr_edge, fut_curr_edge, y_curr_edge]).t()).t()

            # pass through gru
            # unsqueeze to use BatchGRU. Requires 3D inputs.
            h = self.gru(U.t().unsqueeze(0), hx.t().unsqueeze(0)).t().unsqueeze(0)

            eps = self.decoder(h).squeeze(0)

        # return the decoded hx and hx.
        return eps, h

    def initialise_hx_y(self, ys):
        """
        A utility to be accessed by the containing class.
        Initialises the hx and returns them.
        Initialises the hys and stores them.?
        :param ys: observations?
        :return:
        """
        device = ys.device

        if len(ys.shape) == 3:
            batch_size = ys.shape[0]
            if ys.shape[1] == self.y_dim:
                ys = torch.transpose(ys, 1, 2)
            num_samples = ys.shape[1]
            # reorder dimensions to make the next step work properly.
            ys = torch.transpose(ys, 1, 0)
            # concatenate the first element with the rest of the list - end element. transpose
            # again
            y_past = torch.transpose(torch.cat([ys[0].unsqueeze(0), ys[:-1]]), 1, 0)
            y_future = torch.transpose(torch.cat([ys[1:], ys[-1].unsqueeze(0)]), 1, 0)
            # return ys to normal.
            ys = torch.transpose(ys, 1, 0)
        else:
            if ys.shape[0] == self.y_dim:
                # turn from dims x samples to samples x dims
                ys = ys.t()
            num_samples = ys.shape[0]
            y_past = torch.cat([ys[0].unsqueeze(0), ys[:-1]]).t()
            y_future = torch.cat([ys[1:], ys[-1].unsqueeze(0)]).t()
            ys = ys.t()

        # take the difference between past and current ys and current and future ys.
        diff1 = ys - y_past
        diff2 = y_future - ys

        # concatenate and  unsqueeze
        if len(ys.shape) == 3:
            hy_in = torch.cat([diff1, diff2], dim=2).permute(0, 2, 1)
            # save as self.hy
            self.hy = self.hy_initialiser(hy_in)#.permute(1, 2, 0) TODO see if reshaping here is
            # better
            hx = torch.randn((batch_size, self.h_dim, num_samples), device=device)
        else:
            # unsqueeze as conv only takes 3d inputs.
            hy_in = torch.cat([diff1, diff2]).unsqueeze(0)
            # save as self.hy
            self.hy = self.hy_initialiser(hy_in).squeeze(0)
            hx = torch.randn((self.h_dim, num_samples), device=device)

        return hx
