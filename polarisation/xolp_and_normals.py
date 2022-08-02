"""
The script includes the code for both XOLP and normals calculations.
"""

import numpy as np
import cv2
import scipy.interpolate
import matplotlib.pyplot as plt


def split_pol(img):
    """
    :param img: 4x4 polarised image
    :return:
    """
    print(img.shape)
    split_h = np.split(img, 2, axis=1)
    print(split_h[0].shape)

    split_v0 = np.split(split_h[0], 2, axis=0)
    split_v1 = np.split(split_h[1], 2, axis=0)

    pol_00 = split_v0[0]
    pol_10 = split_v0[1]
    pol_01 = split_v1[0]
    pol_11 = split_v1[1]

    return pol_00, pol_10, pol_01, pol_11

def Iun_and_xolp(images, angles):
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

def rho_spec(rho, n):
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

def rho_diffuse(rho, n):
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

def calc_normals(phi, theta):
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
    img_example = cv2.imread("/media/jungo/Research/Datasets/HAMMER/train/scene2_traj1_2/polarization/pol/000000.png", cv2.IMREAD_GRAYSCALE)
    angles = np.array([0, 45, 90, 135]) * np.pi / 180
    n = 1.5

    # print(img_example.shape) # [1664, 2176]
    im00, im10, im01, im11 = split_pol(img_example)
    # print(im0.shape) # [832, 1088]
    im_stack = np.stack((im00, im01, im10, im11), axis=2)  # 0, 45, 90, 135 deg
    # print(im_stack.shape)  # [832, 1088, 4]
    # plt.imshow(im00)
    # plt.show()

    Iun, rho, phi, = Iun_and_xolp(im_stack, angles)
    # print(Iun.shape) # [832, 1088]
    # print(rho.shape) # [832, 1088]
    # print(phi.shape) # [832, 1088]
    # plt.imshow(phi)
    # plt.show()

    theta_diff = rho_diffuse(rho, n)
    theta_spec1, theta_spec2 = rho_spec(rho, n)
    N_diff = calc_normals(phi, theta_diff)
    N_spec1 = calc_normals(phi + np.pi / 2, theta_spec1)
    N_spec2 = calc_normals(phi + np.pi / 2, theta_spec2)

    # print(N_diff.shape) # [832, 1088, 3]
    # print(N_spec1.shape) # [832, 1088, 3]
    # print(N_spec2.shape) # [832, 1088, 3]

if __name__ == "__main__":
    main()
