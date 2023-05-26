# PINN-for-2D-Rayleigh-Bénard-convection

A pytorch implementation of 2D Rayleigh-Bénard convection using PINN

The test case for this repository is Rayleigh Bernard convective flow at Ra=1e8, Pr=2.
DNS data is calculated by the program in the link https://github.com/PhysicsofFluids/AFiD.
Nx=Ny=96, Nt=100;
Lx,Ly=[-0.5, 0.5],[-0.5, 0.5], Lt=20s;

# Note
1. All the data in the wake region is used in this repository. However, if you try different sparsity, you will find it is still trainable.

2. The learning rate scheme used in this repository is that the initial learning rate is set to 1e-3, and it is reduced to 1e-4 after training 100,000 times. This is a more suitable scheme under multiple tests. You can also use other learning rate schemes.

3. Both the data and the equations are non-dimensional, reference Physical Model Schematic Physical_Model_Schematic.png. 

4. If you want to know more about Rayleigh-Bénard convection, please refer to reference 3

5. The code is still being optimized, it would be appreciated to contact me (xinzhiqian@zju.edu.cn) if you find any unreasonably written code.

# Reference
1. Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational physics, 2019, 378: 686-707.
2. Lucor D, Agrawal A, Sergent A. Simple computational strategies for more effective physics-informed neural networks modeling of turbulent natural convection[J]. Journal of Computational Physics, 2022, 456: 111022.
3. Wang Q. Turbulent thermal convection: From Rayleigh-Bénard to vertical convection[J]. 2020.
