import cv2 as cv
import numpy as np

def get_matching_keypoints(template_image, scene_image):
    
    """    
    this fucntion returns the matching keypoints
    from the two images, to use in findhomography function

    """

    # convert the images to grayscale
    template = cv.cvtColor(template_image.copy(), cv.COLOR_BGR2GRAY)
    scene = cv.cvtColor(scene_image.copy(), cv.COLOR_BGR2GRAY)

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
    ratio_thresh = 0.9
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    # draw the good matches
    img_matches = cv.drawMatches(template, kp1, scene, kp2, good_matches, None, flags = 2)

    # display the image
    cv.imshow('Good Matches', img_matches)
    cv.waitKey(0)

    # get the keypoints from the good matches
    object_points = np.zeros((len(good_matches), 2), dtype = np.float32)
    scene_points = np.zeros((len(good_matches), 2), dtype = np.float32)
    for i, match in enumerate(good_matches):
        object_points[i, :] = kp1[match.queryIdx].pt
        scene_points[i, :] = kp2[match.trainIdx].pt

    return object_points, scene_points
