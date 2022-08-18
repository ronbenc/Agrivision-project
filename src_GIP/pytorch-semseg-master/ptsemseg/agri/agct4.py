
import torch
import torch.nn as nn
import numpy as np
import os
from ptsemseg.loader.agrivision_loader_utils import read_agct_coeffs


class agri_color_transform(nn.Module):
    def __init__(self, num_orig_channels = 5, agct_params=None):
        super(agri_color_transform, self).__init__()
        self.num_orig_channels = num_orig_channels
        self.n_channels = agct_params["n_channels"]
        self.lr = agct_params["lr"]
        self.alpha = []
        alpha_coeffs = []
        if "alpha_trained" in agct_params.keys() and os.path.isfile(agct_params["alpha_trained"]):
            alpha_coeffs = read_agct_coeffs(agct_params["alpha_trained"])
            print("agct module - reading pretrained coeffs")
        else:
            alpha_coeffs = agct_params["alpha_coeffs"]
            print("agct module - using initial coeffs")
        for i in range(0,self.n_channels):
            self.alpha.append(torch.tensor(alpha_coeffs[i], requires_grad=True))

    def forward(self, img):
        # self.print_vals('before')
        n1,n2,n3,n4 = img.size()
        CT = torch.zeros(size=[n1,self.n_channels,n3,n4],dtype=float)

        # ~ NDVI
        nomin = self.alpha[0][0] * img[:, 0, :, :] + self.alpha[0][3] * img[:, 3, :, :]
        denom = self.alpha[0][5] * img[:, 0, :, :] + self.alpha[0][8] * img[:, 3, :, :]
        denom[denom==0.0] = 1e-3
        C = nomin / denom
        # C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        CT[:, 0, :, :] = C

        # ~ SAVI
        nomin = self.alpha[1][0] * img[:, 0, :, :] + self.alpha[1][3] * img[:, 3, :, :]
        denom = self.alpha[1][5] * img[:, 0, :, :] + self.alpha[1][8] * img[:, 3, :, :] + self.alpha[1][9]
        denom[denom == 0.0] = 1e-3
        C = nomin / denom
        # C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        CT[:, 1, :, :] = C

        # ~ EVI
        nomin = self.alpha[2][0] * img[:, 0, :, :] + self.alpha[2][3] * img[:, 3, :, :]
        denom = self.alpha[2][5] * img[:, 0, :, :] + self.alpha[2][7] * img[:, 2, :, :] + self.alpha[2][8] * img[:, 3, :, :] + self.alpha[2][9]
        denom[denom == 0.0] = 1e-3
        C = nomin / denom
        # C = torch.nan_to_num(C, nan=0.0, posinf=1.5, neginf=-0.5)   # AZ note extreme cull values
        CT[:, 2, :, :] = C

        # ~ greenNDVI
        nomin = self.alpha[3][1] * img[:, 1, :, :] + self.alpha[3][3] * img[:, 3, :, :]
        denom = self.alpha[3][6] * img[:, 1, :, :] + self.alpha[3][8] * img[:, 3, :, :]
        denom[denom == 0.0] = 1e-3
        C = nomin / denom
        # C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
        CT[:, 3, :, :] = C

        # # ~ GCI
        # nomin = self.alpha[4][1] * img[:, 1, :, :] + self.alpha[4][3] * img[:, 3, :, :]
        # denom = self.alpha[4][6] * img[:, 1, :, :]
        # denom[denom == 0.0] = 1e-3
        # C = nomin / denom
        # # C = torch.nan_to_num(C, nan=0.0, posinf=10.0, neginf=-1.0)  # AZ note extreme cull values
        # CT[:, 4, :, :] = C

        # # ~ SIPI
        # nomin = self.alpha[5][2] * img[:, 2, :, :] + self.alpha[5][3] * img[:, 3, :, :]
        # denom = self.alpha[5][5] * img[:, 0, :, :] + self.alpha[5][8] * img[:, 3, :, :]
        # denom[denom == 0.0] = 1e-3
        # C = 1.0e-3 * (nomin / denom)    # AZ note scaling
        # # C = torch.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)  # AZ note extreme cull values
        # CT[:, 5, :, :] = C


        # others

        # for i in range(5, self.n_channels):
        #     nomin = self.alpha[i][0] * img[:, 0, :, :] + self.alpha[i][1] * img[:, 1, :, :] + self.alpha[i][2] * img[:, 2, :, :] + self.alpha[i][3] * img[:, 3, :, :] + self.alpha[i][4]
        #     denom = self.alpha[i][5] * img[:, 0, :, :] + self.alpha[i][6] * img[:, 1, :, :] + self.alpha[i][7] * img[:, 2, :, :] + self.alpha[i][8] * img[:, 3, :, :] + self.alpha[i][9] + 1e-12
        #     C = nomin / denom
        #     C = torch.nan_to_num(C, nan=0.0)
        #     # outputs[:, i+self.num_orig_channels, :, :] = C
        #     CT[:, i, :, :] = C

        outputs = torch.clone(img)
        outputs[:, self.num_orig_channels:self.num_orig_channels+self.n_channels, :, :] = CT
        # self.print_vals('after')
        return outputs

    def update_weights(self, lr):
        with torch.no_grad():
            for i in range(0,self.n_channels):
                self.alpha[i] -= lr[i] * self.alpha[i].grad
                self.alpha[i].grad.zero_()

    def update_weights_no_grad(self, lr):
        with torch.no_grad():
            for i in range(0,self.n_channels):
                self.alpha[i] -= lr[i] * self.alpha[i].grad

    def update_weights_zero_grad(self):
        with torch.no_grad():
            for i in range(0, self.n_channels):
                self.alpha[i].grad.zero_()

    def get_weights(self):
        with torch.no_grad():
            alpha = np.zeros(shape=(self.n_channels,10), dtype=float)
            alpha_g = np.zeros(shape=(self.n_channels,10), dtype=float)
            for i in range(0, self.n_channels):
                alpha[i,:] = self.alpha[i].detach().numpy()
                alpha_g[i,:] = self.alpha[i].grad.detach().numpy()
            return (alpha, alpha_g)

    def print_vals(self, str_txt):
        print('*************** ' + str_txt)
        for i in range(0, self.n_channels):
            if self.alpha[i] is not None and self.alpha[i].grad is not None:
                alpha = self.alpha[i].detach().numpy()
                alpha_g = self.alpha[i].grad.detach().numpy()
                print('****** ' + str(i) + ' : ')
                print(alpha)
                print(alpha_g)



#
#
# class agri_color_transform(nn.Module):
#     def __init__(self, alpha_n, alpha_d, channel_id):
#         super(agri_color_transform, self).__init__()
#         self.alpha_n = torch.tensor(alpha_n, requires_grad=True)
#         self.alpha_d = torch.tensor(alpha_d, requires_grad=True)
#         self.channel_id = channel_id
#
#     def forward(self, inputs):
#         # self.print_vals('before')
#         nomin = self.alpha_n[0] * inputs[:, 0, : ,:] + self.alpha_n[3] * inputs[: ,3, :, :]
#         denom = self.alpha_d[0] * inputs[:, 0, :, :] + self.alpha_d[3] * inputs[:, 3, :, :] + 1e-12
#         C = nomin / denom
#         C = torch.nan_to_num(C, nan=0.0)
#         outputs = torch.clone(inputs)
#         # self.print_vals('after')
#         outputs[:, self.channel_id, :, :] = C
#         return outputs
#
#     def update_weights(self, lr):
#         with torch.no_grad():
#             self.alpha_n -= lr * self.alpha_n.grad
#             self.alpha_d -= lr * self.alpha_d.grad
#             self.alpha_n.grad.zero_()
#             self.alpha_d.grad.zero_()
#
#     def get_weights(self):
#         nomin = self.alpha_n.detach().numpy()
#         denom = self.alpha_d.detach().numpy()
#         nomin_g = self.alpha_n.grad.detach().numpy()
#         denom_g = self.alpha_d.grad.detach().numpy()
#         return (nomin, denom, nomin_g, denom_g)
#
#     def print_vals(self,str):
#         nomin = self.alpha_n.detach().numpy()
#         denom = self.alpha_d.detach().numpy()
#         print('*************** ' + str)
#         print(nomin)
#         print(denom)
#
