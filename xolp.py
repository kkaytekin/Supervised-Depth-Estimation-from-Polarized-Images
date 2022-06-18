from split_pol import split_pol, stack_pol

import numpy as np
import  cv2
import scipy
import torch
import matplotlib.pyplot as plt



def PolarisationImage_ls(images, angles):
    """
    :param images: 4 concatenate images with different polarisation filters
    :param angles: angles of the polarisation filters
    :return: Iun (unpolarised image), rho (DOLP), phi (AOLP)
    """
    I = images.reshape((images.shape[0] * images.shape[1], 4))
    A = np.zeros((4, 3))
    A[:, 0] = 1
    A[:, 1] = np.cos(2 * angles)
    A[:, 2] = np.sin(2 * angles)
    # x = np.linalg.pinv(A) @ images.T
    x = np.linalg.lstsq(A, I.T, rcond=None)
    x = x[0].T
    Imax = x[:, 0] + np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Imin = x[:, 0] - np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
    Iun = (Imax + Imin) / 2
    # rho = np.divide(Imax - Imin, Imax + Imin)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.true_divide(Imax - Imin, Imax + Imin)
        rho[rho == np.inf] = 0
        rho = np.nan_to_num(rho)
    phi = 0.5 * np.arctan2(x[:, 2], x[:, 1])
    Iun = np.reshape(Iun, (images.shape[0], images.shape[1]))
    rho = np.reshape(rho, (images.shape[0], images.shape[1]))
    phi = np.reshape(phi, (images.shape[0], images.shape[1]))
    return Iun, rho, phi

def rho_spec_ls(rho, n):
    """
    :param rho: DOLP
    :param n: refractive index
    :return: theta_s1, theta_s2 for normals calculation
    """
    theta_s = np.linspace(0, np.pi / 2, 1000)
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
    rho_s1 = rho_s[:imax]
    theta_s1 = theta_s[:imax]
    theta1 = scipy.interpolate.interp1d(rho_s1, theta_s1, fill_value="extrapolate")(rho)

    rho_s2 = rho_s[imax:]
    theta_s2 = theta_s[imax:]
    theta2 = scipy.interpolate.interp1d(rho_s2, theta_s2, fill_value="extrapolate")(rho)
    return theta1, theta2

def rho_diffuse_ls(rho, n):
    """
    :param rho: DOLP
    :param n: refractive index
    :return: theta_d for normals' calculation
    """
    theta_d = np.linspace(0, np.pi / 2, 1000)
    rho_d = ((n - 1 / n) ** 2 * np.sin(theta_d) ** 2) / (
        2
        + 2 * n ** 2
        - (n + 1 / n) ** 2 * np.sin(theta_d) ** 2
        + 4 * np.cos(theta_d) * np.sqrt(n ** 2 - np.sin(theta_d) ** 2)
    )
    theta = scipy.interpolate.interp1d(rho_d, theta_d, fill_value="extrapolate")(rho)
    return theta

def calc_normals_ls(phi, theta):
    """
    :param phi: AOLP
    :param theta: viewing angle (theta_s1 or theta_s2 or theta_d)
    :return: normals (shape: (phi.shape[0], phi.shape[1], 3))
    """
    N1 = np.cos(phi) * np.sin(theta)
    N2 = np.sin(phi) * np.sin(theta)
    N3 = np.cos(theta)
    N = np.zeros((phi.shape[0], phi.shape[1], 3))
    N[:, :, 0] = N1
    N[:, :, 1] = N2
    N[:, :, 2] = N3
    return N


def main():
    img_example = cv2.imread("/home/witek/Documents/Dataset/pol/000000.png", cv2.IMREAD_GRAYSCALE)
    angles = np.array([0, 45, 90, 135]) * np.pi / 180
    n = 1.5

    # img_example = torch.from_numpy(img_example)
    # print(img_example.shape) # [1664, 2176, 1]
    im0, im45, im90, im135 = split_pol(img_example)
    # print(im0.shape) # [832, 1088, 1]
    im_stack = stack_pol(im0, im45, im90, im135) # [832, 1088, 4]
    # print(im_stack.shape)
    # plt.imshow(im0)
    # plt.show()

    Iun, rho, phi, = PolarisationImage_ls(im_stack, angles)
    # print(Iun.shape)
    # print(rho.shape)
    # print(phi.shape)
    # plt.imshow(phi)
    # plt.show()

    # theta_diff = rho_diffuse_ls(rho, n)
    # theta_spec1, theta_spec2 = rho_spec_ls(rho, n)
    # N_diff = calc_normals_ls(phi, theta_diff)
    # N_spec1 = calc_normals_ls(phi + np.pi / 2, theta_spec1)
    # N_spec2 = calc_normals_ls(phi + np.pi / 2, theta_spec2)


if __name__ == "__main__":
    main()