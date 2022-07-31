import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import datetime
import os
from PIL import Image
import cv2

# TODO: define paths
data_path = "data/"
output_path = "output/"
scene = "ar_12_2_2/"

obj_img_path = data_path + scene + "mask_img.png"
depth_gt_img_path = data_path + scene + "depth_gt_img.png"
depth_pred_path = data_path + scene + "depth_pred_img.png"
rgb_img_path = data_path + scene + "rgb_img.png"

logo_path = data_path + "logos/" + "QandA.png"


def print_max_min(image):
    print("Max: " + str(np.max(image)))
    print("Min: " + str(np.min(image)))

def create_naked(show=False):
    """
    create the naked depth image from object mask and depth_gt
    by setting object pixels to max depth in the given depth_gt
    :param show:
    :return:
    """
    obj_naked_img = mpimg.imread(obj_img_path)
    obj_naked_img = obj_naked_img / np.max(obj_naked_img)
    depth_img = mpimg.imread(depth_gt_img_path)

    obj_naked_img[obj_naked_img < 0.3] = 0 # background
    obj_naked_img[obj_naked_img > np.max(obj_naked_img) * 0.9] = 1
    obj_naked_img[obj_naked_img != 0] = 1 #objects bitmap

    for i in range(obj_naked_img.shape[0]):
        #for j in range(obj_naked_img.shape[1]):
        for j in np.invert(range(obj_naked_img.shape[1])):
            if obj_naked_img[i, j] == 1:
                #depth_img[i, j] = depth_img[i, j-1]
                depth_img[i, j] = depth_img[i, j+1]
                #TODO: change the fill direction according to the background

    depth_naked_img = depth_img

    if show:
        plt.figure()
        plt.imshow(depth_naked_img)
        plt.title("depth_naked_img")
        plt.show()

    return depth_naked_img

depth_naked_img_cn = create_naked()

def mask_objects(show=False):
    obj_img = mpimg.imread(obj_img_path)
    obj_naked_img = depth_naked_img_cn
    objs_mask = (obj_img - obj_naked_img)


    objs_mask[objs_mask < 0.3] = 0 #depends on the segmentation of object mask
    objs_mask[objs_mask > 0] = 1

    if show:
        plt.figure()
        plt.imshow(objs_mask)
        plt.title("objs_mask")
        plt.show()

    return objs_mask

def rescale_image(image_path, scale=0.5, show=False):
    img = cv2.imread(image_path, 1)
    if show:
        cv2.imshow('Original', img)
    img_scale_down = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    if show:
        cv2.imshow('Downscaled Image', img_scale_down)
    cv2.waitKey(0)
    return img_scale_down

def ar_mask_init(y_pos, x_pos, logo_pixel_height=100, scale_factor=1.0, show=False):
    """
    bitmap for depth_image with 0 everywhere except where ar object is
    logo_pixel_height: smallest pixel height

    """
    depth_naked_img = depth_naked_img_cn

    logo = mpimg.imread(logo_path)

    depth_value = depth_naked_img[y_pos, x_pos]

    max_depth = np.max(depth_naked_img)

    scale = logo_pixel_height * (max_depth / depth_value)**scale_factor
    scale = scale / logo.shape[0]

    logo_rescaled = rescale_image(logo_path, scale=scale)

    ar_mask = np.zeros_like(depth_naked_img)
    for i in range(logo_rescaled.shape[0]):
        for j in range(logo_rescaled.shape[1]):
            if logo_rescaled[i, j, 2] != 0: # depends on the chosen logo
                ar_mask[y_pos + i, x_pos + j] = 1


    if show:
        plt.figure()
        plt.imshow(ar_mask)
        plt.title("ar_mask")
        plt.show()

    return ar_mask

def get_ar_depth(show=False):
    """ Takes average depth between naked and normal depth images"""

    depth_pred_img = mpimg.imread(depth_pred_path)
    depth_naked_img = depth_naked_img_cn

    depth_ar = depth_naked_img - depth_naked_img * 0.9
    # TODO: test and see for the new scenes

    if show:
        plt.figure()
        plt.imshow(depth_pred_img)
        plt.title("depth_pred_img")
        plt.show()

    objs_mask = mask_objects()
    for i in range(objs_mask.shape[0]):
        for j in range(objs_mask.shape[1]):
            if objs_mask[i, j] == 1:
                depth_ar[i, j] = (depth_pred_img[i, j] + depth_naked_img[i, j]) / 2

    if show:
        plt.figure()
        plt.imshow(depth_ar)
        plt.title("depth_ar: pred + naked /2")
        plt.colorbar()
        plt.show()

    return depth_ar

def put_obj_in_image(obj_y_pos, obj_x_pos,
                     logo_pixel_height=100, scale_factor=1,
                     show=False):
    """
    compare pixelwise depth map of scene with ar object depth values and create binary map,
    init 0, 0 for scene, 1 for square
    """
    depth_pred_img = mpimg.imread(depth_pred_path)
    rgb_img = np.asarray(Image.open(rgb_img_path))

    ar_mask = ar_mask_init(obj_y_pos, obj_x_pos,
                           logo_pixel_height=logo_pixel_height, scale_factor=scale_factor)

    depth_ar = get_ar_depth()
    for i in range(depth_pred_img.shape[0]):
        for j in range(depth_pred_img.shape[1]):
            if ar_mask[i, j] == 1:
                if (depth_pred_img[i, j] > depth_ar[i, j]):
                    rgb_img[i, j, 0] = 0  # R
                    rgb_img[i, j, 1] = 255  # G
                    rgb_img[i, j, 2] = 0  # B # yellow


    if show:
        plt.figure()
        plt.imshow(rgb_img)
        plt.title("rgb_img")
        plt.show()

    return rgb_img



def make_images(num_images_one_way=10,
                 y_start=200, x_start=300,
                 pixel_movement_y=10, pixel_movement_x=10,
                 logo_pixel_height=100, scale_factor=1,
                 show=False):
    image_count = 0
    images = []
    for i in range(num_images_one_way):
        img = put_obj_in_image(y_start + i * pixel_movement_y,
                               x_start + i * pixel_movement_x,
                               logo_pixel_height=logo_pixel_height,
                               scale_factor=scale_factor,
                               show=show)
        img = Image.fromarray(img)
        images.append(img.convert("P", dither=None))
        image_count += 1
        print(image_count)

    return images

def make_ar_gif(images):

    images_rev = images[::-1]
    images_full = images + images_rev

    t = datetime.datetime.now()
    time_path = ("_%s_%s_%s_%s-%s-%s" % (t.year, t.month, t.day, t.hour, t.minute, t.second))
    dir_path = output_path + "depth_check_gif" + time_path
    os.makedirs(dir_path, exist_ok=False)

    images_full[0].save(dir_path + "/depth_check.gif", format="GIF",
                   save_all=True, append_images=images_full[1:],
                   optimize=False, duration=300, loop=0)

if __name__ == "__main__":

    create_naked(show=True)
    mask_objects(show=True)
    ar_mask_init(225, 260, logo_pixel_height=15, scale_factor=1.3, show=True)
    put_obj_in_image(225, 260, logo_pixel_height=15, scale_factor=1.3, show=True)

    images = make_images(num_images_one_way=30,
             y_start=225, x_start=260,
             pixel_movement_y=0, pixel_movement_x=3,
             logo_pixel_height=15, scale_factor=1.3,
             show=True)

    make_ar_gif(images)



























