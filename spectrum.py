import torch.optim
from collections import OrderedDict
import torch.nn as nn
import torch
import os
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


# define nn
class Net(nn.Module):
    def __init__(self, seq_net, name='MLP'):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features['{}_{}'.format(name, i)] = nn.Linear(seq_net[i], seq_net[i + 1], bias=True)
            self.features = nn.ModuleDict(self.features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1:
                break
            i += 1
            act = nn.ELU()
            x = act(x)
        return x


def post_process_average():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用0号GPU
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')  # 没有GPU就用CPU
    PINN = Net(seq_net=[3, 300, 300, 300, 300, 300, 4]).to(device)
    PINN.load_state_dict(torch.load('RB_train_lambda(1600000).pth'))
    for parameters in PINN.parameters():
        print(parameters)
    average_e1 = []

    # read data
    data = io.loadmat('RB_Data.mat')
    x_star = data['X_star']  # N x 2
    t_star = np.arange(0, 2, 2e-2)
    n = x_star.shape[0]
    t_star = np.expand_dims(t_star, axis=1)
    tt = np.tile(t_star, (1, n)).T  # N x T

    for k in range(len(t_star)):
        x1 = x_star[:, 0:1]  # N*1
        y1 = x_star[:, 1:2]  # N*1
        tt1 = tt[:, np.array([k])]
        x_mid = torch.from_numpy(x1).to(device).float()
        y_mid = torch.from_numpy(y1).to(device).float()
        t_mid = torch.from_numpy(tt1).to(device).float()
        with torch.no_grad():
            out = PINN(torch.cat([x_mid, y_mid, t_mid], dim=1))
        u_mid, v_mid = out[:, 0:1], out[:, 1:2]
        u_out = u_mid.cpu().T.detach().numpy().flatten()
        v_out = v_mid.cpu().T.detach().numpy().flatten()
        a = (np.multiply(u_out[0:9409], u_out[0:9409]) + np.multiply(v_out[0:9409], v_out[0:9409])) / 2
        a = np.squeeze(a)
        a = np.sum(a) / 9409
        average_e1.append(a)
        if k % 10 == 0:
            print('{}/100'.format(k))
            print('Ek={}'.format(a))

    path = 'RB_Data.mat'
    data = io.loadmat(path)
    u_start = data['U_star'][:]
    average_e2 = []

    for k in range(len(t_star)):
        a = (np.multiply(u_start[0:9409, 0, k], u_start[0:9409, 0, k])
             + np.multiply(u_start[0:9409, 1, k], u_start[0:9409, 1, k])) / 2
        a = np.squeeze(a)
        a = sum(a) / 9409
        average_e2.append(a)

    return average_e1, average_e2


def fft_plot(e1, e2):
    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    np.seterr(divide='ignore', invalid='ignore')

    t = np.arange(0, 20, 2e-1)  # the real time for DNS{0s、0.2s……20s}
    plt.figure()
    plt.plot(t, e1, 'r')
    plt.scatter(t, e2)
    plt.ylim(0.01, 0.02)
    plt.xlabel('s')
    plt.legend(('PINN', 'DNS'))
    plt.savefig('Turbulent kinetic energy.jpg', dpi=300, bbox_inches='tight')  # save
    plt.show()

    Hz = 5  # fmax for DNS
    size = len(t)
    x = np.arange(size)
    gap = Hz / size

    fft_e1 = fft(e1)
    fft_e2 = fft(e2)
    abs_e1 = np.abs(fft_e1)
    abs_e2 = np.abs(fft_e2)
    # angle_e = np.angle(fft_e)
    normalization_e1 = abs_e1 / (size / 2)
    normalization_e2 = abs_e2 / (size / 2)
    normalization_e1[0] /= 2
    normalization_e2[0] /= 2
    normalization_half_e1 = normalization_e1[range(int(size / 2))]
    normalization_half_e2 = normalization_e2[range(int(size / 2))]
    half_x = x[range(int(size / 2))]

    plt.figure()
    plt.plot(x * gap, abs_e1)
    plt.plot(x * gap, abs_e2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 5e0)
    plt.xlabel('Hz')
    plt.title('spectrum')
    plt.legend(('PINN', 'DNS'))
    plt.savefig('spectrum.jpg', dpi=300, bbox_inches='tight')  # save
    plt.show()

    plt.figure()
    plt.plot(x * gap, normalization_e1)
    plt.plot(x * gap, normalization_e2)
    plt.xscale('log')  # 设置x轴为对数轴
    plt.yscale('log')  # 设置y轴为对数轴
    plt.xlim(1e-3, 5e0)
    plt.xlabel('Hz')
    plt.title('spectrum (normalized)')
    plt.legend(('PINN', 'DNS'))
    plt.savefig('spectrum (normalized).jpg', dpi=300, bbox_inches='tight')  # save
    plt.show()

    plt.figure()
    plt.plot(half_x * gap, normalization_half_e1)
    plt.plot(half_x * gap, normalization_half_e2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 5e0)
    plt.xlabel('Hz')
    plt.title('One-sided spectrum (normalized)')
    plt.legend(('PINN', 'DNS'))
    plt.savefig('One-sided spectrum (normalized).jpg', dpi=300, bbox_inches='tight')  # save
    plt.show()


if __name__ == '__main__':
    e1, e2 = post_process_average()
    fft_plot(e1, e2)
    print('done')
