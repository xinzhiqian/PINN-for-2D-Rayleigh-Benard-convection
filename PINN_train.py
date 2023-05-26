import torch.optim
from collections import OrderedDict
import torch.nn as nn
import torch
import os
import scipy.io as io
import matplotlib.pyplot as plt
from torch.autograd import grad
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')


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


def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def PDE(u_v_p_K, x_f, y_f, t_f, lambda_1, lambda_2):
    u = u_v_p_K[:, 0:1]
    v = u_v_p_K[:, 1:2]
    p = u_v_p_K[:, 2:3]
    K = u_v_p_K[:, 3:]
    out_1 = d(u, x_f) + d(v, y_f)
    out_2 = d(u, t_f) + u * d(u, x_f) + v * d(u, y_f) + \
            d(p, x_f) - pow(lambda_2/lambda_1, 0.5) * (d(d(u, x_f), x_f) + d(d(u, y_f), y_f)) -K
    out_3 = d(v, t_f) + u * d(v, x_f) + v * d(v, y_f) + \
            d(p, y_f) - pow(lambda_2/lambda_1, 0.5) * (d(d(v, x_f), x_f) + d(d(v, y_f), y_f))
    out_4 = d(K, t_f) + u * d(K, x_f) + v * d(K, y_f) - \
            pow(lambda_1 * lambda_2, -0.5) * (d(d(K, x_f), x_f) + d(d(K, y_f), y_f))
    return out_1, out_2, out_3, out_4, u, v, p, K


def train():
    N_train = 10000
    lr = 0.001
    epochs = 1600000
    resample = 1600000  # you may adjust

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    PINN = Net(seq_net=[3, 300, 300, 300, 300, 300, 4]).to(device)
    # PINN.load_state_dict(torch.load('RB_train_lambda(1600000).pth'))
    optimizer = torch.optim.Adam(PINN.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000], gamma=0.1, last_epoch=-1)
    criterion = torch.nn.MSELoss()

    # read data; N=9409, T=100
    data = io.loadmat('RB_Data.mat')
    U_star = data['U_star_new']  # N x 2 x T
    K_star = data['K_star_new']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star_new']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Training Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:], (1, T))  # N X T
    TT = np.tile(t_star, (1, N)).T  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    KK = K_star  # N x T
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    k = KK.flatten()[:, None]  # NT x 1

    # boundary Data
    nx = 97  # grid
    # for uv
    xb = np.zeros((4 * nx, 1))
    xb[0:nx, 0] = -0.5 * np.ones(nx)
    xb[nx:2 * nx, 0] = 0.5 * np.ones(nx)
    xb[2 * nx:3 * nx, 0] = np.arange(-0.5, 0.5, 1 / nx)
    xb[3 * nx:4 * nx, 0] = np.arange(0.5, -0.5, -1 / nx)
    # for k ↑↓
    xb_0 = np.zeros((2 * nx, 1))
    xb_0[0:nx, 0] = -0.5 * np.ones(nx)
    xb_0[nx:2 * nx, 0] = 0.5 * np.ones(nx)
    # for k ←→
    xb_1 = np.zeros((2 * nx, 1))
    xb_1[0:nx, 0] = np.arange(-0.5, 0.5, 1 / nx)
    xb_1[nx:2 * nx, 0] = np.arange(0.5, -0.5, -1 / nx)
    # for uv
    yb = np.zeros((4 * nx, 1))
    yb[0:nx, 0] = np.arange(-0.5, 0.5, 1 / nx)
    yb[nx:2 * nx, 0] = np.arange(0.5, -0.5, -1 / nx)
    yb[2 * nx:3 * nx, 0] = 0.5 * np.ones(nx)
    yb[3 * nx:4 * nx, 0] = -0.5 * np.ones(nx)
    # for k ↑↓
    yb_0 = np.zeros((2 * nx, 1))
    yb_0[0:nx, 0] = np.arange(-0.5, 0.5, 1 / nx)
    yb_0[nx:2 * nx, 0] = np.arange(0.5, -0.5, -1 / nx)
    # for k ←→
    yb_1 = np.zeros((2 * nx, 1))
    yb_1[0:nx, 0] = 0.5 * np.ones(nx)
    yb_1[nx:2 * nx, 0] = -0.5 * np.ones(nx)
    yb_2 = np.zeros((2 * nx, 1))
    yb_2[0:nx, 0] = 0.4974 * np.ones(nx)
    yb_2[nx:2 * nx, 0] = -0.4974 * np.ones(nx)

    x_b = torch.from_numpy(xb).to(device).float().requires_grad_(True)
    x_b_0 = torch.from_numpy(xb_0).to(device).float().requires_grad_(True)
    x_b_1 = torch.from_numpy(xb_1).to(device).float().requires_grad_(True)
    y_b = torch.from_numpy(yb).to(device).float().requires_grad_(True)
    y_b_0 = torch.from_numpy(yb_0).to(device).float().requires_grad_(True)
    y_b_1 = torch.from_numpy(yb_1).to(device).float().requires_grad_(True)
    y_b_2 = torch.from_numpy(yb_2).to(device).float().requires_grad_(True)

    # define the temp boundary condition
    K_boundary = np.zeros((2 * nx, 1))
    K_boundary[0:nx, 0] = 0.5 * np.ones(nx)
    K_boundary[nx:2 * nx, 0] = -0.5 * np.ones(nx)
    K_b = torch.from_numpy(K_boundary).to(device).float().requires_grad_(True)

    # sample for pde and data
    idx = np.random.choice(N * T, N_train, replace=False)  # 从0到N*T中不重复随机抽取N_train个数，并组成1维数组
    x_train = torch.from_numpy(x[idx, :]).to(device).float().requires_grad_(True)  # 计算中保留梯度值
    y_train = torch.from_numpy(y[idx, :]).to(device).float().requires_grad_(True)  # N*1
    t_train = torch.from_numpy(t[idx, :]).to(device).float().requires_grad_(True)
    u_train = torch.from_numpy(u[idx, :]).to(device).float().requires_grad_(True)
    v_train = torch.from_numpy(v[idx, :]).to(device).float().requires_grad_(True)
    k_train = torch.from_numpy(k[idx, :]).to(device).float().requires_grad_(True)
    idt = np.random.choice(T, 1, replace=False)
    t_b = torch.from_numpy(t_star[idt] * np.ones((4 * nx, 1))).to(device).float().requires_grad_(True)
    t_b_0 = torch.from_numpy(t_star[idt] * np.ones((2 * nx, 1))).to(device).float().requires_grad_(True)

    history_loss = []
    torch.cuda.synchronize()
    begin = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        # resample for pde and data
        if epoch % resample == 0:
            idx = np.random.choice(N * T, N_train, replace=False)
            x_train = torch.from_numpy(x[idx, :]).to(device).float().requires_grad_(True)
            y_train = torch.from_numpy(y[idx, :]).to(device).float().requires_grad_(True)
            t_train = torch.from_numpy(t[idx, :]).to(device).float().requires_grad_(True)
            u_train = torch.from_numpy(u[idx, :]).to(device).float().requires_grad_(True)
            v_train = torch.from_numpy(v[idx, :]).to(device).float().requires_grad_(True)
            k_train = torch.from_numpy(k[idx, :]).to(device).float().requires_grad_(True)
            idt = np.random.choice(T, 1, replace=False)
            t_b = torch.from_numpy(t_star[idt] * np.ones((4 * nx, 1))).to(device).float().requires_grad_(True)
            t_b_0 = torch.from_numpy(t_star[idt] * np.ones((2 * nx, 1))).to(device).float().requires_grad_(True)

        # PDE and Data Loss
        u_v_p_K = PINN(torch.cat([x_train, y_train, t_train], dim=1))
        PDE_1, PDE_2, PDE_3, PDE_4, u_u, u_v, u_p, u_k = PDE(u_v_p_K, x_train, y_train, t_train, 1e8, 2)  # Ra、Pr

        mse_PDE_1 = criterion(PDE_1, torch.zeros_like(PDE_1))
        mse_PDE_2 = criterion(PDE_1, torch.zeros_like(PDE_2))
        mse_PDE_3 = criterion(PDE_3, torch.zeros_like(PDE_3))
        mse_PDE_4 = criterion(PDE_4, torch.zeros_like(PDE_4))
        mse_PDE = mse_PDE_2 + mse_PDE_1 + mse_PDE_3 + mse_PDE_4
        mse_Data_1 = criterion(u_u, u_train)
        mse_Data_2 = criterion(u_v, v_train)
        mse_Data_3 = criterion(u_k, k_train)
        mse_Data = mse_Data_1 + mse_Data_2 + mse_Data_3

        # Boundary Loss
        # uv
        u_v_p_K_b_1 = PINN(torch.cat([x_b, y_b, t_b], dim=1))
        b_u, b_v = u_v_p_K_b_1[:, 0:1], u_v_p_K_b_1[:, 1:2]
        # k ↑↓
        u_v_p_K_b_2 = PINN(torch.cat([x_b_0, y_b_0, t_b_0], dim=1))
        b_K_2 = u_v_p_K_b_2[:, 3:4]
        # k ←→
        u_v_p_K_b_3 = PINN(torch.cat([x_b_1, y_b_1, t_b_0], dim=1))
        b_K_3 = u_v_p_K_b_3[:, 3:4]
        u_v_p_K_b_4 = PINN(torch.cat([x_b_1, y_b_2, t_b_0], dim=1))
        b_K_4 = u_v_p_K_b_4[:, 3:4]
        # First-order difference to approximate the temp gradient
        b_K_34 = torch.div((b_K_3 - b_K_4), 1 / (nx - 1))  # 1/96

        mse_Boundary_1 = criterion(b_u, torch.zeros_like(b_u)) + criterion(b_v, torch.zeros_like(b_v))
        mse_Boundary_2 = criterion(b_K_2, K_b)
        mse_Boundary_3 = criterion(b_K_34, torch.zeros_like(b_K_34))
        mse_Boundary = mse_Boundary_1 + mse_Boundary_2 + mse_Boundary_3

        # loss
        loss = 1 * mse_PDE + 1 * mse_Data + 1 * mse_Boundary

        # print loss
        if epoch % 1000 == 0:
            print(
                'epoch:{:05d}, PDE_loss: {:.08e}, Data_loss: {:.08e}, '
                'Boundary_loss: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_Data.item(), mse_Boundary.item(), loss.item()
                )
            )

        history_loss.append([mse_PDE.item(), mse_Data.item(), mse_Boundary.item(), loss.item()])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print time and lr
        if (epoch + 1) % 1000 == 0:
            torch.cuda.synchronize()
            end = time.time()
            print('time for 1000 epoch = {:} s'.format(end - begin))  # print time
            torch.cuda.synchronize()
            begin = time.time()
            # print lr
            print(optimizer.state_dict()['param_groups'][0]['lr'])

    # plotting loss
    plt.plot(history_loss)
    plt.legend(('PDE loss', 'Data loss', 'Boundary loss', 'loss'))
    plt.yscale('log')
    plt.ylim(1e-6, 1e-0)
    plt.savefig('loss{}'.format(epochs))
    plt.show()
    torch.save(PINN.state_dict(), 'RB_train_lambda({}).pth'.format(epochs))

    return


if __name__ == '__main__':

    train()
    print('done')



