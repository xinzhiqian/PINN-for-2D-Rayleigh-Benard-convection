import torch.optim
from collections import OrderedDict
import torch.nn as nn
import torch
import os
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import imageio
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


def plot(t):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    PINN = Net(seq_net=[3, 300, 300, 300, 300, 300, 4]).to(device)
    PINN.load_state_dict(torch.load('RB_train_lambda(1600000).pth'))

    # read data; N=9409, T=100
    data = io.loadmat('RB_Data.mat')
    U_star = data['U_star']  # N x 2 x T
    K_star = data['K_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    # rearrange
    TT = np.tile(t_star, (1, N)).T  # N x T

    for it in range(0, t, 10):
        snap = np.array([it])  # 1*1
        x_star = X_star[:, 0:1]  # N*1
        y_star = X_star[:, 1:2]  # N*1
        t_star = TT[:, snap]  # N*1
        u_star = U_star[:, 0, snap]  # N*1*1
        v_star = U_star[:, 1, snap]  # N*1*1
        k_star = K_star[:, snap]  # N*1

        # Test data
        x_test = torch.from_numpy(x_star).to(device).float().requires_grad_(True)
        y_test = torch.from_numpy(y_star).to(device).float().requires_grad_(True)
        t_test = torch.from_numpy(t_star).to(device).float().requires_grad_(True)
        u_test = torch.from_numpy(u_star).to(device).float()
        v_test = torch.from_numpy(v_star).to(device).float()
        k_test = torch.from_numpy(k_star).to(device).float()

        # grid
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 97  # nx
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        # The xy swap is because the xy axis is the opposite of the actual situation when generating DNS data
        Y, X = np.meshgrid(x, y)

        judg = None
        if judg is None:
            with torch.no_grad():
                u_v_p_K = PINN(torch.cat([x_test, y_test, t_test], dim=1))
            u_pred, v_pred, p_pred, k_pred = u_v_p_K[:, 0:1], u_v_p_K[:, 1:2], u_v_p_K[:, 2:3], u_v_p_K[:, 3:4]
            error_u, error_v, error_k = abs(u_pred - u_test), abs(v_pred - v_test), abs(k_pred - k_test)
            # print loss for test
            mse = torch.nn.MSELoss()
            mse_u = mse(u_pred, u_test)
            mse_v = mse(v_pred, v_test)
            mse_k = mse(k_pred, k_test)
            mse_test = mse_u + mse_v + mse_k
            print('Data_test_loss = {:.08e}'.format(mse_test.item()))
            UU_star = griddata(X_star, u_pred.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')
            VV_star = griddata(X_star, v_pred.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')
            KK_star = griddata(X_star, k_pred.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')
            UU_star_error = griddata(X_star, error_u.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')
            VV_star_error = griddata(X_star, error_v.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')
            KK_star_error = griddata(X_star, error_k.cpu().T.detach().numpy().flatten(), (X, Y), method='cubic')

            # pred u
            plt.imshow(UU_star, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            plt.title('u-PINN(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            cbar_u = plt.colorbar()
            cbar_u.mappable.set_clim(-0.5, 0.5)
            plt.savefig('({}).u.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()

            plt.imshow(UU_star_error, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            cbar_u_error = plt.colorbar()
            cbar_u_error.mappable.set_clim(0, 0.5)
            plt.title('u-error(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('({}).u_error.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()

            # pred v
            plt.imshow(VV_star, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            plt.title('v-PINN(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            cbar_v = plt.colorbar()
            cbar_v.mappable.set_clim(-0.5, 0.5)
            plt.savefig('({}).v.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()

            plt.imshow(VV_star_error, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            plt.title('v-error(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            cbar_v_error = plt.colorbar()
            cbar_v_error.mappable.set_clim(0, 0.5)
            plt.savefig('({}).v_error.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()

            # pred k
            plt.imshow(KK_star, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            plt.title('k-PINN(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            cbar_k = plt.colorbar()
            cbar_k.mappable.set_clim(-0.5, 0.5)
            plt.savefig('({}).k.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()

            plt.imshow(KK_star_error, interpolation='nearest', cmap='rainbow',
                       extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                       origin='lower', aspect='auto')
            plt.title('k-error(t={:.1f}s)'.format(it * 0.2))
            plt.xlabel('x')
            plt.ylabel('y')
            cbar_k_error = plt.colorbar()
            cbar_k_error.mappable.set_clim(0, 0.5)
            plt.savefig('({}).k_error.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
            plt.close()


def action_fig(num):
    #
    with imageio.get_writer(uri='action_v_pred.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).v.jpg'))
    with imageio.get_writer(uri='action_v_error.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).v_error.jpg'))

    with imageio.get_writer(uri='action_u_pred.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).u.jpg'))
    with imageio.get_writer(uri='action_u_error.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).v_error.jpg'))

    with imageio.get_writer(uri='action_k_pred.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).k.jpg'))
    with imageio.get_writer(uri='action_k_error.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).k_error.jpg'))


if __name__ == '__main__':

    t_sum = 100
    plot(t_sum)
    action_fig(t_sum)
    print('done')
