import numpy as np
import torch
import scipy.interpolate


def rho_diffuse(rho, n):
    rho = rho.cpu().numpy()  # BxHxW
    theta_d = np.linspace(0, np.pi / 2, 1000)  # vector
    rho_d = ((n - 1 / n) ** 2 * np.sin(theta_d) ** 2) / (
        2
        + 2 * n ** 2
        - (n + 1 / n) ** 2 * np.sin(theta_d) ** 2
        + 4 * np.cos(theta_d) * np.sqrt(n ** 2 - np.sin(theta_d) ** 2)
    )  # vector
    f = scipy.interpolate.interp1d(rho_d, theta_d, fill_value="extrapolate")
    theta = torch.from_numpy(f(rho))
    return theta


def rho_spec(rho, n):
    rho = rho.cpu().numpy()  # BxHxW
    theta_s = np.linspace(0, np.pi / 2, 1000)  # vector
    rho_s = (
        2
        * np.sin(theta_s) ** 2
        * np.cos(theta_s)
        * np.sqrt(n ** 2 - np.sin(theta_s) ** 2)
    ) / (
        n ** 2
        - np.sin(theta_s) ** 2
        - n ** 2 * np.sin(theta_s) ** 2
        + 2 * np.sin(theta_s) ** 4
    )
    imax = np.argmax(rho_s)
    rho_s1 = rho_s[:imax]  # vector
    theta_s1 = theta_s[:imax]  # vector
    f1 = scipy.interpolate.interp1d(rho_s1, theta_s1, fill_value="extrapolate")
    theta1 = torch.from_numpy(f1(rho))

    rho_s2 = rho_s[imax:]  # vector
    theta_s2 = theta_s[imax:]  # vector
    f2 = scipy.interpolate.interp1d(rho_s2, theta_s2, fill_value="extrapolate")
    theta2 = torch.from_numpy(f2(rho))

    return theta1, theta2


def calc_normals(phi, theta):
    phi = phi.cuda()
    theta = theta.cuda()
    N1 = (torch.cos(phi) * torch.sin(theta)).unsqueeze(dim=1)  # Bx1xHxW
    N2 = (torch.sin(phi) * torch.sin(theta)).unsqueeze(dim=1)  # Bx1xHxW
    N3 = torch.cos(theta).unsqueeze(dim=1)  # Bx1xHxW
    #print("N3: ", N3.shape)
    N = torch.cat((N1, N2, N3), dim=1)  # Bx3xHxW
    return N
