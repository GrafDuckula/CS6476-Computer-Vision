"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy import ndimage


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    return np.sqrt((p0[0]-p1[0])**2+(p0[1]-p1[1])**2)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    # print (image)
    # print (image.shape)
    top_left = (0,0)
    bottom_left = (0,image.shape[0]-1)
    top_right = (image.shape[1]-1,0)
    bottom_right = (image.shape[1]-1,image.shape[0]-1)
    
    return [top_left, bottom_left, top_right, bottom_right]


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    if template is None:
        template_gray = np.ones((3,3))/9.0
        
    img = np.copy(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    image_norm = img_gray.astype(float)-128
    template_norm = template_gray.astype(float)-128
    
    
    # part 1 and 2a
    # convolution
#    img_con = cv2.filter2D(image_norm, -1, template_norm)
#    img_con_norm = (img_con-np.min(img_con))*256/(np.max(img_con)-np.min(img_con))
    
    # search for the top 4 brightest point
    # part 1
#    top_four = np.argpartition(img_con_norm, -4, axis=None)[-4:]
#    width = img_con.shape[1]
#    idx = [divmod(i, width) for i in top_four]
    
    #    print (idx)
#    for i in idx:
#        cv2.circle(img, (i[1], i[0]), 1, (0, 0, 255), 1)
        
    # part 2a
#    tops = np.argpartition(img_con_norm, -100, axis=None)[-100:]
#    width = img_con.shape[1]
#    idxs = [divmod(i, width) for i in tops]
#    
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#    # (type,max_iter,epsilon)
#    flags=cv2.KMEANS_RANDOM_CENTERS
#    compactness,labels,centers = cv2.kmeans(np.float32(idxs), 4, None, criteria, 10, flags)
#    
##    print (centers)
#    for i in centers:
#        cv2.circle(img, (i[1], i[0]), 1, (0, 0, 255), 1)
#    
#    idx = centers
    
    
    # part 2 with rotation
    
    best = 0
    idx = []
    for deg in range(0, 180, 10):
        template_rot = ndimage.rotate(template_norm, deg, reshape=False)
    
        img_con = cv2.filter2D(image_norm, -1, template_rot)
        img_con_norm = (img_con-np.min(img_con))*256/(np.max(img_con)-np.min(img_con))

        # tops = np.argpartition(img_con_norm, -120, axis=None)[-120:]

        # for 4 and 5
        # tops = np.argpartition(img_con_norm, -290, axis=None)[-290:]

        # for c of 4 and 5
        tops = np.argpartition(img_con_norm, -320, axis=None)[-320:]

        # for 3-a-1
        # tops = np.argpartition(img_con_norm, -225, axis=None)[-225:]

        width = img_con.shape[1]
        idxs = [divmod(i, width) for i in tops]
    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        # (type,max_iter,epsilon)
        flags=cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(np.float32(idxs), 4, None, criteria, 10, flags)
        
        brightness_li = [img_con_norm[int(i[0]), int(i[1])] for i in centers]
        brightness = np.average(brightness_li)
        
        # if one distance is much shorter than the other two, increase the top number?
        
        
        
        
        if brightness > best:
            best = brightness
            idx = centers
            
    for i in idx:
        cv2.circle(img, (i[1], i[0]), 1, (0, 0, 255), 1)
    
    
    
#    cv2.imshow('input image',image)
#    cv2.imshow('normalized_image',image_norm.astype('uint8'))
#    cv2.imshow('filtered_image',img_con_norm.astype('uint8'))
#    cv2.imshow('template_image',template.astype('uint8'))
#    cv2.imshow('template_image_norm',template_norm.astype('uint8'))
#    cv2.imshow('marked image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    
    # find relative position of each marker
    sorted_left = sorted(idx, key=lambda tup: tup[1])[:2]
    sorted_right = sorted(idx, key=lambda tup: tup[1])[2:]
    top_left = sorted(sorted_left, key=lambda tup: tup[0])[0]
    bottom_left = sorted(sorted_left, key=lambda tup: tup[0])[1]
    top_right = sorted(sorted_right, key=lambda tup: tup[0])[0]
    bottom_right = sorted(sorted_right, key=lambda tup: tup[0])[1]  
    
    # swap x and y
    top_left = (int(top_left[1]),int(top_left[0])) 
    bottom_left = (int(bottom_left[1]),int(bottom_left[0])) 
    top_right = (int(top_right[1]),int(top_right[0])) 
    bottom_right = (int(bottom_right[1]),int(bottom_right[0]))
    
#    print ([top_left, bottom_left, top_right, bottom_right])
    
    
    return [top_left, bottom_left, top_right, bottom_right]
    


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    
    cv2.line(image, markers[0], markers[1], (0,0,255), thickness)
    cv2.line(image, markers[1], markers[3], (0,0,255), thickness)
    cv2.line(image, markers[3], markers[2], (0,0,255), thickness)
    cv2.line(image, markers[2], markers[0], (0,0,255), thickness)
    
    return image


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    # corners = find_markers(imageB)
    # print (corners)
    # [top_left, bottom_left, top_right, bottom_right]
    # row_min = min(corners[0][1], corners[2][1])
    # row_max = max(corners[1][1], corners[3][1])
    # col_min = min(corners[0][0], corners[1][0])
    # col_max = max(corners[2][0], corners[3][0])
    #
    # maps = np.transpose(homography)*imageB[row_min:row_max, col_min:col_max]

    r, c = imageB.shape[:2]
    ind_y, ind_x = np.indices((r, c), dtype=np.float32)
    linear_hom_ind = np.array([ind_x.ravel(), ind_y.ravel(), np.ones_like(ind_x).ravel()])
    # print (lin_homg_ind)

    map = np.linalg.inv(homography).dot(linear_hom_ind)
    # print (map_ind)
    map_x, map_y = map[:-1] / map[-1]
    map_x = map_x.reshape(r, c).astype(np.float32)
    map_y = map_y.reshape(r, c).astype(np.float32)

    imageB = cv2.remap(imageA, map_x, map_y, cv2.INTER_LINEAR, dst=imageB, borderMode=cv2.BORDER_TRANSPARENT)

    return imageB


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    a = []
    for i in range(len(src_points)):
        x = src_points[i][0]
        y = src_points[i][1]
        x_p = dst_points[i][0]
        y_p = dst_points[i][1]
        a.append([-x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p])
        a.append([0, 0, 0, -x, -y, -1, x*y_p,y*y_p, y_p])

    a = np.asarray(a)
    u, s, vh = np.linalg.svd(a, full_matrices=True)

    # print (vh)
    # print (vh[-1, :])
    # print (np.reshape(vh[-1, :], (3, 3)))
    H = np.reshape(vh[-1, :], (3, 3))
    H_norm = H/H[2,2]
    # print (H_norm)

    return H_norm


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
