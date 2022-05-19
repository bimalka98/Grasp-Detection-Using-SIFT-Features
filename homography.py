import cv2 as cv
import numpy as np

# fucntion to get the matching feature points
def get_matching_keypoints(template_image, scene_image):
    
    """    
    this fucntion returns the matching keypoints
    from the two images, to use in findhomography function

    """

    # convert the images to grayscale
    template = cv.cvtColor(template_image.copy(), cv.COLOR_BGR2GRAY)
    scene = cv.cvtColor(scene_image.copy(), cv.COLOR_BGR2GRAY)

    # gaussian blur the images: to remove noice
    template = cv.GaussianBlur(template, (3, 3), 0)
    scene = cv.GaussianBlur(scene, (3, 3), 0)

    # detect features using SIFT and compute the descriptors
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(scene, None)

    # matching descriptors using FLANN matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # filter matches using the Lowe's ratio test
    ratio_thresh = 0.85
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print("Number of matches:", len(good_matches))

    # sort the matches based on the distance
    good_matches = sorted(good_matches, key = lambda x:x.distance)

    # draw the good matches
    img_matches = cv.drawMatches(template, kp1, scene, kp2, good_matches, None, flags = 2)

    # display the image
    cv.imshow('Good Matches', img_matches)
    cv.waitKey(0)

    # --------removing the matches that are not in the target area--------
    # Method 1 of the paper.
    #  
    # optimization parameters
    threashold_numof_neighbors = 4 # number of neighbors around the point of interest
    width_of_target_area = template.shape[1]/4
    height_of_target_area = template.shape[0]/4

    # array to store the filtered matches
    correct_matches = []

    # iterate over the matches
    for match in good_matches:
        
        # the point of interest
        point_of_interest = kp1[match.queryIdx].pt 
        
        # number of neighbors in the inspection area
        num_neighbors = 0 

        # iterate over the matches again to find the neighbors
        for neighbor in good_matches:
            
            # the neighbor point under inspection
            neighbor_point = kp1[neighbor.queryIdx].pt 

            # get absolute distance between the two points
            delta_x = abs(neighbor_point[0] - point_of_interest[0])
            delta_y = abs(neighbor_point[1] - point_of_interest[1])

            # if the distance is less than the inspection area, add the point to the list
            if delta_x < width_of_target_area and delta_y < height_of_target_area:
                num_neighbors += 1

        # if the number of neighbors is greater than the threashold, add the point to the list
        if num_neighbors >= threashold_numof_neighbors:
            correct_matches.append(match)
    
    print("Number of correct matches:", len(correct_matches))

    # draw the correct matches
    correct_feature_matches = cv.drawMatches(template, kp1, scene, kp2, correct_matches, None, flags = 2)
    cv.imshow('Correct Matches', correct_feature_matches)
    cv.waitKey(0)

    # --------------------------------------------------------------------

    # get the keypoints from the good matches
    object_points = np.zeros((len(correct_matches), 2), dtype = np.float32)
    scene_points = np.zeros((len(correct_matches), 2), dtype = np.float32)
    for i, match in enumerate(correct_matches):
        object_points[i, :] = kp1[match.queryIdx].pt
        scene_points[i, :] = kp2[match.trainIdx].pt

    
    return object_points, scene_points

# fucntion to remove the mismatching points: outliers
def remove_mismatches(object_points, scene_points):

    """
    RANSAC algorithms can be improved using the following two methods
    presented in the paper: 

    "SIFT Feature Point Matching Based on Improved RANSAC Algorithm"
    by Guangjun Shi, Xiangyang Xu, Yaping Dai

    1. Method One: Find the target area and remove the
    feature points not in the target area.

    2. Method Two: Remove the crossing feature points: this can not 
    be used here as the images may be roatated in extreme degrees.

    THE SAME METHOD IS IMPLEMENTED INSIDE THE FUNCTION: get_matching_keypoints()
    """

    # Method 1:
#     It is supposed that feature point A(x, y) is one of the
# matching feature points. Calculate the number of matching
# feature points in the rectangular region whose center
# is (x, y) , width is “a” and height is “b” (“a” and “b” are set
# by us). If the number less than the threshold value, A(x, y)
# is supposed the isolated point not in the target area. Then
# remove A(x, y) . Examine each of the matching points.

    correct_matches = 0
    filtered_object_points = []
    filtered_scene_points = []

    # set the threshold value
    threshold = 3 # number of matching neighbors around the point of interest

    # set the width and height of the target area
    width = 60 # width of the target area in pixels
    height = 60 # height of the target area in pixels

    # iterate through the object points
    for i in range(len(object_points)):

        object_point = object_points[i]
        scene_point = scene_points[i]

        # calculate the number of matching neighbors around the point of interest
        num_neighbors = 0
        for neighbor in object_points:
            delta_x = abs(neighbor[0] - object_point[0])
            delta_y = abs(neighbor[1] - object_point[1])

            # if the neighbor is in the target area
            if delta_x < width/2 and delta_y < height/2:
                num_neighbors += 1
            
        # if the number of matching neighbors is greater than the threshold value,
        # the point is in the target area
        if num_neighbors >= threshold:
            correct_matches += 1
            filtered_object_points.append(object_point)
            filtered_scene_points.append(scene_point)

    # convert the filtered points to numpy arrays
    filtered_object_points2 = np.zeros((correct_matches, 2), dtype = np.float32)
    filtered_scene_points2 = np.zeros((correct_matches, 2), dtype = np.float32)
    for i in range(correct_matches):
        filtered_object_points2[i, :] = filtered_object_points[i]
        filtered_scene_points2[i, :] = filtered_scene_points[i]
    
    print("Number of correct matches: ", correct_matches)
    return filtered_object_points2, filtered_scene_points2

# function to calculate transformed points
def get_transformed_grasp_locations(grasp_locations, homography_matrix):

    """
    get the transformed grasp locations on the scene image.
    """

    # make grasp_locations a numpy array
    grasp_locations = np.array(grasp_locations)
    # print(grasp_locations)

    # add a column of ones to grasp_locations to make it homogeneous
    grasp_locations = np.concatenate((grasp_locations, np.ones((grasp_locations.shape[0], 1))), axis=1)

    # transform the grasp_locations using the homography matrix
    transformed_grasp_locations = np.matmul(homography_matrix, grasp_locations.T) 
    # print(transformed_grasp_locations)

    # divide the transformed_grasp_locations by the last column to get the final grasp locations
    transformed_grasp_locations = transformed_grasp_locations / transformed_grasp_locations[-1, :]

    # return the final grasp locations
    transformed_grasp_locations = transformed_grasp_locations[:-1, :].T
    
    return transformed_grasp_locations
