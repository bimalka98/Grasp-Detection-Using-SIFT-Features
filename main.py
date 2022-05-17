import cv2 as cv
import numpy as np

from io_helper import * 
from homography import *

def main(object_class):

    # get the template image and the scene images form the directory
    template_image = cv.imread('template/' + object_class + '.jpg', cv.IMREAD_ANYCOLOR)
    scene_image = cv.imread('scene/' + object_class + '.jpg', cv.IMREAD_ANYCOLOR)

    # get the grasping location from the user as mouse clicked points
    grasp_locations = get_grasp_locations(template_image)

    # print the mouse clicked points
    print(grasp_locations)

    # get matching keypoints from the template and scene images
    object_points, scene_points = get_matching_keypoints(template_image, scene_image)

    # get the homography matrix
    homography_matrix, _ = cv.findHomography(object_points, scene_points, cv.RANSAC)
    print("Homography Matrix : \n", homography_matrix)

    # get the transformed grasp locations on the scene image







if __name__ == '__main__':

    # print the opencv version
    print("OpenCV version : ", cv.__version__)

    # execute the main fucntion     
    main(input("Enter the object class : ").strip())