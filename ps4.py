"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1.0/8)
    return sobelx


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8)
    return sobely


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    if k_type == "uniform":
        img_a_blur = cv2.blur(img_a, (k_size, k_size))
        img_b_blur = cv2.blur(img_b, (k_size, k_size))
    elif k_type == "gaussian":
        img_a_blur = cv2.GaussianBlur(img_a, (k_size, k_size),sigma)
        img_b_blur = cv2.GaussianBlur(img_b, (k_size, k_size),sigma)

    sobelx = gradient_x(img_a_blur)
    sobely = gradient_y(img_a_blur)
    sobelt = img_b_blur-img_a_blur

    IxIx = np.square(sobelx)
    IyIy = np.square(sobely)
    IxIy = np.multiply(sobelx, sobely)
    IxIt = np.multiply(sobelx, sobelt)
    IyIt = np.multiply(sobely, sobelt)

    # k_win = 37 # code
    k_win = k_size

    sum_IxIx = cv2.boxFilter(IxIx, cv2.CV_64F, (k_win, k_win), normalize=False)
    sum_IxIy = cv2.boxFilter(IxIy, cv2.CV_64F, (k_win, k_win), normalize=False)
    sum_IyIy = cv2.boxFilter(IyIy, cv2.CV_64F, (k_win, k_win), normalize=False)
    sum_IxIt = cv2.boxFilter(IxIt, cv2.CV_64F, (k_win, k_win), normalize=False)
    sum_IyIt = cv2.boxFilter(IyIt, cv2.CV_64F, (k_win, k_win), normalize=False)

    ATA = np.array([[sum_IxIx,sum_IxIy],[sum_IxIy,sum_IyIy]]).transpose((2,3,0,1))
    det = np.linalg.det(ATA)
    ATA_inv = np.array([[sum_IyIy,-sum_IxIy],[-sum_IxIy,sum_IxIx]]).transpose((2,3,0,1))/det[:,:,None,None]

    ATb = np.array([[-sum_IxIt], [-sum_IyIt]]).transpose((2, 3, 0, 1))

    UV = np.matmul(ATA_inv, ATb)

    # print (ATA.shape)
    # print (ATb.shape)
    #
    # print (UV.shape)
    # print (UV[:,:,0, 0].shape)
    # print (UV[:, :, 1, 0].shape)

    # return (UV[:,:,0, 0], UV[:,:,1, 0])

    return (cv2.GaussianBlur(UV[:,:,0, 0], (k_size, k_size),sigma), cv2.GaussianBlur(UV[:,:,1, 0], (k_size, k_size),sigma))


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    # 1 4 6 4 1
    kernel = np.array([[1, 4, 6, 4, 1]])/16.0
    reduce_x = cv2.filter2D(image, cv2.CV_64F, kernel)
    reduce_y = cv2.filter2D(reduce_x, cv2.CV_64F, np.transpose(kernel))

    # print (image)
    # print ()
    # print (reduce_x)
    # print()
    # print (reduce_y)
    # print()
    # print (image.shape)
    # print (reduce_y[::2,::2].shape)

    return reduce_y[::2,::2]


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    img = np.copy(image)
    image_list = []

    for i in range(levels):
        image_list.append(img)
        img = reduce_image(img)

    return image_list


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    row_num = img_list[0].shape[0]
    col_num = 0

    for img in img_list:
        col_num += img.shape[1]

    output = np.zeros((row_num, col_num))

    col_pos = 0
    for img in img_list:
        img_norm = normalize_and_scale(img)
        start_pos = col_pos
        col_pos += img.shape[1]
        row_pos = img.shape[0]
        output[:row_pos, start_pos:col_pos] = img_norm

    return output


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    row_num = image.shape[0]
    col_num = image.shape[1]
    output = np.zeros((row_num*2, col_num*2))
    output[::2, ::2] = image

    kernel = np.array([[1, 4, 6, 4, 1]])/8.0
    expand_x = cv2.filter2D(output, cv2.CV_64F, kernel)
    expand_y = cv2.filter2D(expand_x, cv2.CV_64F, np.transpose(kernel))

    return expand_y


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    l_pyr = []
    for i in range(len(g_pyr)-1):
        expanded = expand_image(g_pyr[i+1])

        diff_row = expanded.shape[0] - g_pyr[i].shape[0]
        diff_col = expanded.shape[1] - g_pyr[i].shape[1]

        if diff_row > 0:
            expanded = expanded[:-diff_row,:]

        if diff_col > 0:
            expanded = expanded[:, :-diff_col]

        laplacian = g_pyr[i]-expanded
        l_pyr.append(laplacian)

    l_pyr.append(g_pyr[-1])

    return l_pyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    A = np.zeros((image.shape[0], image.shape[1]))
    M, N = A.shape
    X, Y = np.meshgrid(range(N), range(M))

    # (X+U, Y+V)
    map1 = (X + U).astype(np.float32)
    map2 = (Y + V).astype(np.float32)

    imageB = cv2.remap(image, map1, map2, interpolation, borderMode=border_mode)

    return imageB


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    reduce_list_a = []
    reduce_list_b = []
    reduce_a = np.copy(img_a)
    reduce_b = np.copy(img_b)

    for i in range(levels):
        reduce_list_a.append(reduce_a)
        reduce_list_b.append(reduce_b)

        reduce_a = reduce_image(reduce_a)
        reduce_b = reduce_image(reduce_b)

    img_a_LK = reduce_list_a[-1]

    u_expand = np.zeros_like(img_a_LK)
    v_expand = np.zeros_like(img_a_LK)

    for i in range(levels-1, -1, -1):
        # print (i)
        u, v = optic_flow_lk(img_a_LK, reduce_list_b[i], k_size, k_type, sigma=sigma)

        u = u + u_expand
        v = v + v_expand

        if i != 0:
            u_expand = expand_image(u)
            v_expand = expand_image(v)

            diff_u = u_expand.shape[0] - reduce_list_a[i-1].shape[0]
            diff_v = v_expand.shape[1] - reduce_list_a[i-1].shape[1]

            if diff_u > 0:
                u_expand = u_expand[:-diff_u, :]
                v_expand = v_expand[:-diff_u, :]

            if diff_v > 0:
                u_expand = u_expand[:, :-diff_v]
                v_expand = v_expand[:, :-diff_v]

            img_a_LK = warp(reduce_list_a[i-1], -1*u_expand, -1*v_expand, interpolation, border_mode)
        #
        # print (u.shape)
        # print (v.shape)

        # print (u)
        # print (v)


    return (u, v)
