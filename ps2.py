"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import math
import numpy as np


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    
    # Can I do this?


    img = np.copy(img_in)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
    # img_blur = cv2.medianBlur(img_gray,5)

    # part 4
    img_blur = cv2.medianBlur(img, 9)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # for part 1
    radii_range = range(10, 50, 1)
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, \
    #                            minDist=img.shape[0]/20, \
    #                            param1=30, param2=20, \
    #                            minRadius=min([i for i in radii_range]),\
    #                            maxRadius=max([i for i in radii_range]))

    # for part 3
    # radii_range = range(3, 40, 1)
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, \
    #                            minDist=10, \
    #                            param1=100, param2=20, \
    #                            minRadius=min([i for i in radii_range]),\
    #                            maxRadius=max([i for i in radii_range]))


    # for part 4
    radii_range = range(5, 25, 1)
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, \
                               minDist=10, \
                               param1=40, param2=18, \
                               minRadius=min([i for i in radii_range]),\
                               maxRadius=max([i for i in radii_range]))
    # 873 197 161 125
    result = ((0,0),"Yellow")

    if circles is None:
        # print ("None")
        return result
    # print (circles)

    for i in circles[0][:]:
        cv2.circle(img_blur, (i[0], i[1]), i[2], (255, 255, 255), 1)
        cv2.circle(img_blur, (i[0], i[1]), 1, (255, 255, 255), 1)
    
    sorted_circle = sorted(circles[0,:], key=lambda x:x[1])
    
    median_circle = sorted_circle[int(np.floor(len(sorted_circle)/2))]
    tol = 10
    clean_circles = []
    for i in range(len(sorted_circle)):
        if abs(sorted_circle[i][0]-median_circle[0])<tol:
            clean_circles.append(sorted_circle[i])

    if len(clean_circles) < 3:
        return result

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    # (type,max_iter,epsilon)
    flags=cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(np.float32(clean_circles), 3, None, criteria, 10, flags)
#    data, K, criteria, attempts, flags
#     np.float32
    
    for i in clean_circles[:]:
        cv2.circle(img_blur, (i[0],i[1]),i[2],(255,255,255),1)
        cv2.circle(img_blur, (i[0],i[1]),1,(255,255,255),1)

    if len(centers) == 3:
        # coor of yellow light

        print (centers[:,:2])
        coors = sorted(centers[:,:2], key=lambda x:x[1])
        center_coor = sorted(centers[:,:2], key=lambda x:x[1])[1]
        # sort small to big, red-yellow-green

    #    detect traffic light color
        tol = 10
        colors = ["red", "yellow", "green"]
        color = "default"
        for i in range(3):
            row = int(np.floor(coors[i][1]))
            col = int(np.floor(coors[i][0]))
            # if abs(max(img_in[row,col,:])-255)<=tol:
            #     color = colors[i]

            # part 4
            if abs(max(img_blur[row,col,:])-255)<=tol:
                color = colors[i]
        result = (tuple([int(center_coor[0]), int(center_coor[1])]), color)
    
    # cv2.imshow('input image',img)
    # cv2.imshow('detected circles',img_blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return result

def null(x):
    pass
    
def traffic_light_detection_w_trackbar(img_in, radii_range):
    
    # Can I do this?
    radii_range = range(3, 40, 1)
    
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img_gray, (5,5),0)
    img = cv2.medianBlur(img_gray,5)
    cv2.namedWindow("ParamsTune")
    cv2.createTrackbar("param1", "ParamsTune", 100, 200, null)
    cv2.createTrackbar("param2", "ParamsTune", 30, 200, null)
    cv2.createTrackbar("minDist", "ParamsTune", 10, 200, null)
    cv2.createTrackbar("dp", "ParamsTune", 1, 5, null)
    
    while(1):
        key = cv2.waitKey(1)&0xFF
        # print (key)
        if key == 7:
            break
        param1 = cv2.getTrackbarPos("param1", "ParamsTune")
        param2 = cv2.getTrackbarPos("param2", "ParamsTune")
        minDist = cv2.getTrackbarPos("minDist", "ParamsTune")
        dp = cv2.getTrackbarPos("dp", "ParamsTune")
    
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, \
                                   minDist=minDist, \
                                   param1=param1, param2=param2, \
                                   minRadius=min([i for i in radii_range]),\
                                   maxRadius=max([i for i in radii_range]))
        if circles is None:
            return ((0,0),"Yellow")
        # print(circles)
        for i in circles[:]:
            cv2.circle(img, (i[0][0], i[0][1]), i[0][2], (255, 255, 255), 1)
            cv2.circle(img, (i[0][0], i[0][1]), 1, (255, 255, 255), 1)
    
        cv2.imshow('detected circles',img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Not real
    coor = (2,3)
    color = 'yellow'

    return (coor,color)


def get_intersections(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return x0, y0


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img = np.copy(img_in)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
    # img_blur = cv2.medianBlur(img_gray,5)
    # img_edge = cv2.Canny(img_blur, 50, 200)

    img_blur = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 200)


    # lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 70)
    # part 3
    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 50)

    # part 4
    # lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 40)

    centers = []
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            pt1 = (int(math.cos(theta) * rho + 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho + 2000 * (math.cos(theta))))
            pt2 = (int(math.cos(theta) * rho - 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho - 2000 * (math.cos(theta))))

            cv2.line(img_blur, pt1, pt2, (255, 255, 255), 1)

        # get lines with the right angles, 30, 90, 150 degree
#        tol = 0.1
        
        # part 5
        tol = 0.2
        angle1 = list(filter(lambda x: abs(x[0][1] - 2.618) <= tol, lines))
        angle2 = list(filter(lambda x: abs(x[0][1] - 1.571) <= tol, lines))
        angle3 = list(filter(lambda x: abs(x[0][1] - 0.524) <= tol, lines))

        # get the intersections of lines and check colors
        intersections = []
        tol = 10
        for i in range(len(angle1)):
            for j in range(len(angle2)):
                for k in range(len(angle3)):
                    rho1, theta1 = angle1[i][0]
                    rho2, theta2 = angle2[j][0]
                    rho3, theta3 = angle3[k][0]

                    x0, y0 = get_intersections(rho1, theta1, rho2, theta2)
                    x1, y1 = get_intersections(rho1, theta1, rho3, theta3)
                    x2, y2 = get_intersections(rho2, theta2, rho3, theta3)

                    # print(x0, y0, x1, y1, x2, y2)

                    # 5% length inside the triangle is red color.
                    length = x2 - x0
                    check_point = [int((x0 + x2)/2), int((y0 + y2)/2 + length*0.05)]

                    print (check_point)

                    # check color
                    # if (abs(img[check_point[1], check_point[0], 0] - 0) <= tol and
                    #         abs(img[check_point[1], check_point[0], 1] - 0) <= tol and
                    #         abs(img[check_point[1], check_point[0], 2] - 255) <= tol):
                    #     intersections.append([[x0, y0], [x1, y1], [x2, y2]])

                    # part 4
#                    if (abs(img_blur[check_point[1], check_point[0], 0] - 0) <= tol and
#                            abs(img_blur[check_point[1], check_point[0], 1] - 0) <= tol and
#                            abs(img_blur[check_point[1], check_point[0], 2] - 255) <= tol):
#                        intersections.append([[x0, y0], [x1, y1], [x2, y2]])
                    # part 5    
                    if (abs(img_blur[check_point[1], check_point[0], 0] - 180) <= tol and
                            abs(img_blur[check_point[1], check_point[0], 1] - 180) <= tol and
                            abs(img_blur[check_point[1], check_point[0], 2] - 180) <= tol):
                        intersections.append([[x0, y0], [x1, y1], [x2, y2]])                    


        # print (intersections)

        if intersections is not None:
            for i in range(len(intersections)):
                centerX = (intersections[i][0][0] + intersections[i][1][0] + intersections[i][2][0])/3
                centerY = (intersections[i][0][1] + intersections[i][1][1] + intersections[i][2][1]) / 3

                centers.append([centerX, centerY])

    x_sum = 0
    y_sum = 0
    weight_center = (0, 0)

    if len(centers) != 0:
        for i in range(len(centers)):
            x_sum += centers[i][0]
            y_sum += centers[i][1]

        weight_center=(int(x_sum/len(centers)), int(y_sum/len(centers)))

        cv2.circle(img_blur, (weight_center[0], weight_center[1]), 1, (0, 0, 0), 3)

    cv2.imshow('input image', img_in)
    cv2.imshow('edges', img_edge)
    cv2.imshow('detected lines', img_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print (weight_center)

    return weight_center



def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    img = np.copy(img_in)

    # part 2
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
    # img_blur = cv2.medianBlur(img_gray,5)
    # img_edge = cv2.Canny(img_blur, 50, 200)
    # lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 30)

    # part 3 4 noise
    img_blur = cv2.medianBlur(img, 9)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 200)

#    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 35)
    
    # part 5
    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 35)

#    print (lines)

    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
    
            pt1 = (int(math.cos(theta) * rho + 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho + 2000 * (math.cos(theta))))
            pt2 = (int(math.cos(theta) * rho - 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho - 2000 * (math.cos(theta))))
    
            cv2.line(img, pt1, pt2, (255, 255, 255), 1)

    # part 1 2 3
    tol = 0.1
    # part 4
    # tol = 0.05
    angle1 = list(filter(lambda x: abs(x[0][1] - 0.785) <= tol, lines))
    angle2 = list(filter(lambda x: abs(x[0][1] - 2.356) <= tol, lines))

#    print (angle1)
#    print (angle2)
         
    # get the intersections of lines and check colors

#    cv2.imshow('input image', img)
#    cv2.imshow('edges', img_edge)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    intersections = []
#    tol = 10
    # part 5
    tol = 40
    centers = []
    for i in range(len(angle1)):
        for j in range(len(angle2)):
            for k in range(i+1,len(angle1)):
                for l in range(j+1,len(angle2)):
                    rho1, theta1 = angle1[i][0]
                    rho2, theta2 = angle2[j][0]
                    rho3, theta3 = angle1[k][0]
                    rho4, theta4 = angle2[l][0]
        
                    x0, y0 = get_intersections(rho1, theta1, rho2, theta2)
                    x1, y1 = get_intersections(rho2, theta2, rho3, theta3)
                    x2, y2 = get_intersections(rho3, theta3, rho4, theta4)
                    x3, y3 = get_intersections(rho1, theta1, rho4, theta4)
        
                    # print (x0, y0, x1, y1, x2, y2, x3, y3)
                    
                    d1 = cv2.norm(np.float32([x0, y0]), np.float32([x1, y1]), cv2.NORM_L2)
                    d2 = cv2.norm(np.float32([x1, y1]), np.float32([x2, y2]), cv2.NORM_L2)
                    d3 = cv2.norm(np.float32([x2, y2]), np.float32([x3, y3]), cv2.NORM_L2)
                    d4 = cv2.norm(np.float32([x3, y3]), np.float32([x0, y0]), cv2.NORM_L2)
                    
                    side_lengths = np.array([d1, d2, d3, d4])

                    min_x = min(x0, x1, x2, x3)
                    min_y = min(y0, y1, y2, y3)
                    
                    center_x = np.mean([x0, x1, x2, x3])
                    center_y = np.mean([y0, y1, y2, y3])

                    check_point = [int(min_x+0.35*np.mean(side_lengths)), int(min_y + 0.35 * np.mean(side_lengths))]

                    # print (img.shape)
                    if check_point[0]>=img.shape[1] or check_point[1]>=img.shape[0]:
                        continue
                    # if y1 < y0:
                    #     check_point = [x1, int(y1+0.35*np.mean(side_lengths))]
                    # elif y1 == y0:
                    #     continue
                    # else:
                    #     check_point = [x1, int(y1-0.35*np.mean(side_lengths))]

                    # print (check_point)
                    
                    # the side lengths are at most 5% difference
                    if (abs(max(side_lengths)-min(side_lengths))<=0.1*np.mean(side_lengths) and 
                        abs(max(side_lengths)-min(side_lengths))>0):

                        # check color to be red 204
                        
                        # if (abs(img[check_point[1], check_point[0], 0] - 0) <= tol and
                        #     abs(img[check_point[1], check_point[0], 1] - 0) <= tol and
                        #     abs(img[check_point[1], check_point[0], 2] - 204) <= tol):
                        if (abs(img_blur[check_point[1], check_point[0], 0] - 0) <= tol and
                            abs(img_blur[check_point[1], check_point[0], 1] - 0) <= tol and
                            abs(img_blur[check_point[1], check_point[0], 2] - 204) <= tol):
                        
                            intersections.append([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                            
                            centers.append([int(center_x),int(center_y)])       
#                            print (check_point)
                    
#    print (intersections)
    sum_x = 0
    sum_y = 0
    for c in centers:
        sum_x += c[0]
        sum_y += c[1]   
    
    weight_center = (0, 0)
    if len(centers) != 0:
        weight_center = (int(sum_x/len(centers)), int(sum_y/len(centers)))
#    print (weight_center)
    
#    cv2.imshow('input image', img_in)
#    cv2.imshow('edges', img_edge)
#    cv2.imshow('detected lines', img_blur)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    return weight_center



def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # yellow square
    
    img = np.copy(img_in)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
    # img_blur = cv2.medianBlur(img_gray,5)
    # img_edge = cv2.Canny(img_blur, 50, 200)

    # part 4 noise
    img_blur = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 200)

#    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 50)
    
    # part 5
    lines = cv2.HoughLines(img_edge, 1, np.pi/180, 70)
    

#    print (lines)
    centers = []
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            
            pt1 = (int(math.cos(theta) * rho + 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho + 2000 * (math.cos(theta))))
            pt2 = (int(math.cos(theta) * rho - 2000 * (-math.sin(theta))), \
                   int(math.sin(theta) * rho - 2000 * (math.cos(theta))))
            
            cv2.line(img, pt1, pt2, (255, 255, 255), 1)
        
        
        tol = 0.1
        angle1 = list(filter(lambda x: abs(x[0][1] - 0.785) <= tol, lines))
        angle2 = list(filter(lambda x: abs(x[0][1] - 2.356) <= tol, lines))

    #    print (angle1)
    #    print (angle2)

        # get the intersections of lines and check colors
        intersections = []
        tol = 10
        # part 5a
        tol = 35
        for i in range(len(angle1)):
            for j in range(len(angle2)):
                for k in range(i+1,len(angle1)):
                    for l in range(j+1,len(angle2)):
                        rho1, theta1 = angle1[i][0]
                        rho2, theta2 = angle2[j][0]
                        rho3, theta3 = angle1[k][0]
                        rho4, theta4 = angle2[l][0]

                        x0, y0 = get_intersections(rho1, theta1, rho2, theta2)
                        x1, y1 = get_intersections(rho2, theta2, rho3, theta3)
                        x2, y2 = get_intersections(rho3, theta3, rho4, theta4)
                        x3, y3 = get_intersections(rho1, theta1, rho4, theta4)

                        # print(x0, y0, x1, y1, x2, y2, x3, y3))

                        d1 = cv2.norm(np.float32([x0, y0]), np.float32([x1, y1]), cv2.NORM_L2)
                        d2 = cv2.norm(np.float32([x1, y1]), np.float32([x2, y2]), cv2.NORM_L2)
                        d3 = cv2.norm(np.float32([x2, y2]), np.float32([x3, y3]), cv2.NORM_L2)
                        d4 = cv2.norm(np.float32([x3, y3]), np.float32([x0, y0]), cv2.NORM_L2)

                        side_lengths = np.array([d1, d2, d3, d4])
                        check_point = [x1, y0]
                        ave_side_length = np.mean(side_lengths)
                        
                        if check_point[0]>=img.shape[1] or check_point[1]>=img.shape[0]:
                            continue                        
                        # the side lengths are at most 5% difference
                        if (abs(max(side_lengths)-min(side_lengths))<=0.1*np.mean(side_lengths) and
                            abs(max(side_lengths)-min(side_lengths))>0):
                            # check color to be yellow
                            # if (abs(img[check_point[1], check_point[0], 0] - 0) <= tol and
                            #     abs(img[check_point[1], check_point[0], 1] - 255) <= tol and
                            #     abs(img[check_point[1], check_point[0], 2] - 255) <= tol):
                            #
                            #     c1 = check_color(img, (int(check_point[0] + ave_side_length*0.5), check_point[1]),
                            #                      (0, 255, 255), tol)
                            #     c2 = check_color(img, (int(check_point[0] - ave_side_length * 0.5), check_point[1]),
                            #                      (0, 255, 255), tol)
                            #     c3 = check_color(img, (check_point[0], int(check_point[1] + ave_side_length * 0.5)),
                            #                      (0, 255, 255), tol)
                            #     c4 = check_color(img, (check_point[0], int(check_point[1] - ave_side_length * 0.5)),
                            #                      (0, 255, 255), tol)
                            if (abs(img_blur[check_point[1], check_point[0], 0] - 0) <= tol and
                                abs(img_blur[check_point[1], check_point[0], 1] - 255) <= tol and
                                abs(img_blur[check_point[1], check_point[0], 2] - 255) <= tol):

                                c1 = check_color(img_blur, (int(check_point[0] + ave_side_length*0.5), check_point[1]),
                                                 (0, 255, 255), tol)
                                c2 = check_color(img_blur, (int(check_point[0] - ave_side_length * 0.5), check_point[1]),
                                                 (0, 255, 255), tol)
                                c3 = check_color(img_blur, (check_point[0], int(check_point[1] + ave_side_length * 0.5)),
                                                 (0, 255, 255), tol)
                                c4 = check_color(img_blur, (check_point[0], int(check_point[1] - ave_side_length * 0.5)),
                                                 (0, 255, 255), tol)

                                if c1 and c2 and c3 and c4:
                                    intersections.append([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                                    centers.append(check_point)
#                                    print (check_point)
                    
#    print (intersections)
    sum_x = 0
    sum_y = 0

    for c in centers:
        sum_x += c[0]
        sum_y += c[1]

    weight_center = (0, 0)
    if len(centers) != 0:
        weight_center = (int(sum_x/len(centers)), int(sum_y/len(centers)))
#    print (weight_center)
    
#    cv2.imshow('input image', img_in)
#    cv2.imshow('edges', img_edge)
#    cv2.imshow('detected lines', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    return weight_center

def check_color(img, point, color, tol):
    if (abs(img[point[1], point[0], 0] - color[0]) <= tol and
            abs(img[point[1], point[0], 1] - color[1]) <= tol and
            abs(img[point[1], point[0], 2] - color[2]) <= tol):
        return True
    else:
        return False




def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
        # orange square
    
    img = np.copy(img_in)

    # # part 123
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
    # img_blur = cv2.medianBlur(img_gray,5)
    # img_edge = cv2.Canny(img_blur, 50, 200)
    # #    img_edge = cv2.Canny(img_blur, 10, 300)

    # part 4 noise
    img_blur = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 50, 200)

    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 50)

#    print (lines)
    centers = []
    if lines is not None:
        # for i in range(len(lines)):
        #     rho = lines[i][0][0]
        #     theta = lines[i][0][1]
        #
        #     pt1 = (int(math.cos(theta) * rho + 2000 * (-math.sin(theta))), \
        #            int(math.sin(theta) * rho + 2000 * (math.cos(theta))))
        #     pt2 = (int(math.cos(theta) * rho - 2000 * (-math.sin(theta))), \
        #            int(math.sin(theta) * rho - 2000 * (math.cos(theta))))
        #
        #     cv2.line(img_blur, pt1, pt2, (255, 255, 255), 1)

        tol = 0.1
        angle1 = list(filter(lambda x: abs(x[0][1] - 0.785) <= tol, lines))
        angle2 = list(filter(lambda x: abs(x[0][1] - 2.356) <= tol, lines))

    #    print (angle1)
    #    print (angle2)

        # get the intersections of lines and check colors
        intersections = []
        tol = 10
        
        for i in range(len(angle1)):
            for j in range(len(angle2)):
                for k in range(i+1,len(angle1)):
                    for l in range(j+1,len(angle2)):
                        rho1, theta1 = angle1[i][0]
                        rho2, theta2 = angle2[j][0]
                        rho3, theta3 = angle1[k][0]
                        rho4, theta4 = angle2[l][0]

                        x0, y0 = get_intersections(rho1, theta1, rho2, theta2)
                        x1, y1 = get_intersections(rho2, theta2, rho3, theta3)
                        x2, y2 = get_intersections(rho3, theta3, rho4, theta4)
                        x3, y3 = get_intersections(rho1, theta1, rho4, theta4)

                        # print(x0, y0, x1, y1, x2, y2, x3, y3))

                        d1 = cv2.norm(np.float32([x0, y0]), np.float32([x1, y1]), cv2.NORM_L2)
                        d2 = cv2.norm(np.float32([x1, y1]), np.float32([x2, y2]), cv2.NORM_L2)
                        d3 = cv2.norm(np.float32([x2, y2]), np.float32([x3, y3]), cv2.NORM_L2)
                        d4 = cv2.norm(np.float32([x3, y3]), np.float32([x0, y0]), cv2.NORM_L2)

                        side_lengths = np.array([d1, d2, d3, d4])
                        check_point = [x1, y0]
                        ave_side_length = np.mean(side_lengths)
                        
                        if check_point[0]>=img.shape[1] or check_point[1]>=img.shape[0]:
                            continue
                        # the side lengths are at most 5% difference

                        if (abs(max(side_lengths)-min(side_lengths))<=0.1*np.mean(side_lengths) and
                            abs(max(side_lengths)-min(side_lengths))>0):
                            # check color to be orange
                            # if (abs(img[check_point[1], check_point[0], 0] - 0) <= tol and
                            #     abs(img[check_point[1], check_point[0], 1] - 127) <= tol and
                            #     abs(img[check_point[1], check_point[0], 2] - 255) <= tol):
                            #
                            #     c1 = check_color(img, (int(check_point[0] + ave_side_length*0.5), check_point[1]),
                            #                      (0, 127, 255), tol)
                            #     c2 = check_color(img, (int(check_point[0] - ave_side_length * 0.5), check_point[1]),
                            #                      (0, 127, 255), tol)
                            #     c3 = check_color(img, (check_point[0], int(check_point[1] + ave_side_length * 0.5)),
                            #                      (0, 127, 255), tol)
                            #     c4 = check_color(img, (check_point[0], int(check_point[1] - ave_side_length * 0.5)),
                            #                      (0, 127, 255), tol)
                            if (abs(img_blur[check_point[1], check_point[0], 0] - 0) <= tol and
                                abs(img_blur[check_point[1], check_point[0], 1] - 127) <= tol and
                                abs(img_blur[check_point[1], check_point[0], 2] - 255) <= tol):

                                # print ("inner cycle")
                                # print (check_point)
                                #
                                # print (int(check_point[0] + ave_side_length*0.4), check_point[1])
                                # print (int(check_point[0] - ave_side_length * 0.4), check_point[1])
                                # print (check_point[0], int(check_point[1] + ave_side_length * 0.4))
                                # print (check_point[0], int(check_point[1] - ave_side_length * 0.4))


                                c1 = check_color(img_blur, (int(check_point[0] + ave_side_length*0.5), check_point[1]),
                                                 (0, 127, 255), tol)
                                c2 = check_color(img_blur, (int(check_point[0] - ave_side_length * 0.5), check_point[1]),
                                                 (0, 127, 255), tol)
                                c3 = check_color(img_blur, (check_point[0], int(check_point[1] + ave_side_length * 0.5)),
                                                 (0, 127, 255), tol)
                                c4 = check_color(img_blur, (check_point[0], int(check_point[1] - ave_side_length * 0.5)),
                                                 (0, 127, 255), tol)

                                # print (c1, c2, c3, c4)
                                # print ([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

                                if c1 and c2 and c3 and c4:
                                    intersections.append([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                                    centers.append(check_point)
                                    # print (check_point)
                    
#    print (intersections)
    sum_x = 0
    sum_y = 0
    weight_center = (0, 0)

    for c in centers:
        sum_x += c[0]
        sum_y += c[1]   
    
    weight_center = (0, 0)
    if len(centers) != 0:
        weight_center = (int(sum_x/len(centers)), int(sum_y/len(centers)))
#    print (weight_center)
    
    # cv2.imshow('input image', img_in)
    # cv2.imshow('edges', img_edge)
    # cv2.imshow('detected lines', img_blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return weight_center



def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    # radii_range = range(10, 100, 1)
    radii_range = range(10, 50, 1)
    img = np.copy(img_in)

#     img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
# #    img_blur = cv2.GaussianBlur(img_gray, (5,5),0)
#     img_blur = cv2.medianBlur(img_gray,5)

    img_blur = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # part 1 2
    # circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1, \
    #                            minDist=img_blur.shape[0]/20, \
    #                            param1=30, param2=20, \
    #                            minRadius=min([i for i in radii_range]),\
    #                            maxRadius=max([i for i in radii_range]))
    # part 3
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, \
                               minDist=img_blur.shape[0]/20, \
                               param1=30, param2=20, \
                               minRadius=min([i for i in radii_range]),\
                               maxRadius=max([i for i in radii_range]))

    # part 4
    # circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, \
    #                            minDist=img_blur.shape[0]/20, \
    #                            param1=100, param2=30, \
    #                            minRadius=min([i for i in radii_range]),\
    #                            maxRadius=max([i for i in radii_range]))

    # print (circles)
    if circles is None:
        return (0,0)
    tol = 30
    clean_circles = []
    for i in range(len(circles[0])):
        # if (abs(img[int(circles[0][i][1]),int(circles[0][i][0]), 0] - 255) <= tol and
        #         abs(img[int(circles[0][i][1]),int(circles[0][i][0]), 1] - 255) <= tol and
        #         abs(img[int(circles[0][i][1]),int(circles[0][i][0]), 2] - 255) <= tol):
        if (abs(img_blur[int(circles[0][i][1]),int(circles[0][i][0]), 0] - 255) <= tol and
                abs(img_blur[int(circles[0][i][1]),int(circles[0][i][0]), 1] - 255) <= tol and
                abs(img_blur[int(circles[0][i][1]),int(circles[0][i][0]), 2] - 255) <= tol):

            clean_circles.append(circles[0][i])
    
    for i in clean_circles[:]:
        cv2.circle(img_blur, (i[0],i[1]),i[2],(255,255,255),1)
        cv2.circle(img_blur, (i[0],i[1]),1,(255,255,255),1)

    # print (clean_circles)
    sum_x = 0
    sum_y = 0
    for i in range(len(clean_circles)):
        sum_x += clean_circles[i][0]
        sum_y += clean_circles[i][1]

    center = (0, 0)
    if len(clean_circles) != 0:
        center = (int(sum_x/len(clean_circles)), int(sum_y/len(clean_circles)))

    # print (center)

    cv2.imshow('input image',img_in)
    cv2.imshow('detected circles',img_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return center


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    radii_range = range(10, 40, 1)
    signs = {}
    stp_coords = stop_sign_detection(img_in)
    constr_coords = construction_sign_detection(img_in)
    wrng_coords = warning_sign_detection(img_in)
    dne_coords = do_not_enter_sign_detection(img_in)
    yld_coords = yield_sign_detection(img_in)
    trf_coords, state = traffic_light_detection(img_in, radii_range)

    if stp_coords != (0,0):
       signs['stop'] = stp_coords

    if constr_coords != (0,0):
       signs['construction'] = constr_coords

    if wrng_coords != (0,0):
       signs['warning'] = wrng_coords
    #
    if dne_coords != (0,0):
       signs['no_entry'] = dne_coords
    #
    if yld_coords != (0,0):
        signs['yield'] = yld_coords

    if trf_coords != (0,0):
        signs['traffic_light'] = trf_coords

    return signs


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    radii_range = range(10, 100, 1)
    signs = {}
    stp_coords = stop_sign_detection(img_in)
    constr_coords = construction_sign_detection(img_in)
    wrng_coords = warning_sign_detection(img_in)
    dne_coords = do_not_enter_sign_detection(img_in)
    yld_coords = yield_sign_detection(img_in)
    trf_coords, state = traffic_light_detection(img_in, radii_range)



    if stp_coords != (0, 0):
        signs['stop'] = stp_coords

    if constr_coords != (0, 0):
        signs['construction'] = constr_coords

    if wrng_coords != (0, 0):
        signs['warning'] = wrng_coords

    if dne_coords != (0, 0):
        signs['no_entry'] = dne_coords

    if yld_coords != (0, 0):
        signs['yield'] = yld_coords

    if trf_coords != (0, 0):
        signs['traffic_light'] = trf_coords

    return signs


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    radii_range = range(10, 100, 1)
    signs = {}
    stp_coords = stop_sign_detection(img_in)
#    constr_coords = construction_sign_detection(img_in)
#    wrng_coords = warning_sign_detection(img_in)
#    dne_coords = do_not_enter_sign_detection(img_in)
#    yld_coords = yield_sign_detection(img_in)
#    trf_coords, state = traffic_light_detection(img_in, radii_range)



    if stp_coords != (0, 0):
        signs['stop'] = stp_coords
#
#    if constr_coords != (0, 0):
#        signs['construction'] = constr_coords
#
#    if wrng_coords != (0, 0):
#        signs['warning'] = wrng_coords
#
#    if dne_coords != (0, 0):
#        signs['no_entry'] = dne_coords

#    if yld_coords != (0, 0):
#        signs['yield'] = yld_coords

#    if trf_coords != (0, 0):
#        signs['traffic_light'] = trf_coords

    return signs


#    yellow = np.array([0,127,127])
#    green = np.array([0,127,0])
#    red = np.array([0,0,127])
#    orange = np.array([0, 127, 255])
    
#    for i in circles[0,:]:
#        cv2.circle(img, (i[0],i[1]),i[2],(255,255,255),1)
#        cv2.circle(img, (i[0],i[1]),1,(255,255,255),1)


    # linesP = cv2.HoughLinesP(img_blur, 1, np.pi / 180, 800, 50, 2)
    #
    # if linesP is not None:
    #     for i in range(len(linesP)):
    #         line = linesP[i][0]
    #         cv2.line(img_blur, (line[0],line[1]), (line[2],line[3]), (255,255,255),1)