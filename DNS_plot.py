# Exact for DNS：
import scipy.io as io
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import imageio


def Exact(t):
    # read data
    data = io.loadmat('RB_Data.mat')
    U_star = data['U_star']  # N x 2 x T
    K_star = data['K_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    # test Data
    snap = np.array([t])
    x_star = X_star[:, 0:1]  # Nx1
    y_star = X_star[:, 1:2]  # Nx1
    # grid
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 97
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    # The xy swap is because the xy axis is the opposite of the actual situation when generating DNS data
    Y, X = np.meshgrid(x, y)  #

    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    k_star = K_star[:, snap]

    U_exact = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    V_exact = griddata(X_star, v_star.flatten(), (X, Y), method='cubic')
    K_exact = griddata(X_star, k_star.flatten(), (X, Y), method='cubic')
    plt.imshow(U_exact, interpolation='nearest', cmap='rainbow',
               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
               origin='lower', aspect='auto')
    plt.title('u-DNS(t={:.1f}s)'.format(it * 0.2))
    plt.xlabel('x')
    plt.ylabel('y')
    cbar_k = plt.colorbar()
    cbar_k.mappable.set_clim(-0.5, 0.5)
    plt.savefig('({}).u_dns.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
    plt.close()

    plt.imshow(V_exact, interpolation='nearest', cmap='rainbow',
               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
               origin='lower', aspect='auto')
    plt.title('v-DNS(t={:.1f}s)'.format(it * 0.2))
    plt.xlabel('x')
    plt.ylabel('y')
    cbar_k = plt.colorbar()
    cbar_k.mappable.set_clim(-0.5, 0.5)
    plt.savefig('({}).v_dns.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
    plt.close()

    # temp
    plt.imshow(K_exact, interpolation='nearest', cmap='rainbow',
               extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
               origin='lower', aspect='auto')
    plt.title('k-DNS(t={:.1f}s)'.format(it * 0.2))
    plt.xlabel('x')
    plt.ylabel('y')
    cbar_k = plt.colorbar()
    cbar_k.mappable.set_clim(-0.5, 0.5)
    plt.savefig('({}).k_dns.jpg'.format(it), dpi=300, bbox_inches='tight')  # save
    plt.close()


def action_fig():
    num = 100
    # action fig v
    with imageio.get_writer(uri='action_v_dns.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).v_dns.jpg'))
    # action fig u
    with imageio.get_writer(uri='action_u_dns.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).u_dns.jpg'))
    # action fig k
    with imageio.get_writer(uri='action_k_dns.gif', mode='I', fps=2) as writer:
        for i in range(0, num, 10):
            writer.append_data(imageio.imread(f'({i}).k_dns.jpg'))


if __name__ == '__main__':

    t_sum = 100
    for t_plot in range(0, t_sum, 10):
        Exact(t_plot)
    print('done')

