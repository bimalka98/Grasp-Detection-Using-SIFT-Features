import cv2 as cv
import numpy as np

from io_helper import * 
from homography import *

# main fucntion to be executed
def main(object_class):

    # get the template image and the scene images form the directory
    template_image = cv.imread('template/' + object_class + '.jpg', cv.IMREAD_ANYCOLOR)
    scene_image = cv.imread('scene/' + object_class + '.jpg', cv.IMREAD_ANYCOLOR)

    # get the grasping location from the user as mouse clicked points
    grasp_locations = get_grasp_locations(template_image)

    # get matching keypoints from the template and scene images
    object_points, scene_points = get_matching_keypoints(template_image, scene_image)

    # get the homography matrix
    if len(object_points) >= 4:
        
        print("Finding the homography matrix...")

        # find the homography matrix: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html 
        # cv.findHomography https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780    
        homography_matrix, _ = cv.findHomography(object_points, scene_points, cv.RANSAC) # cv.RHO # cv.LMEDS        
        print("Homography Matrix : \n", homography_matrix)

        # get the transformed grasp locations on the scene image
        try:
            transformed_grasp_locations = get_transformed_grasp_locations(grasp_locations, homography_matrix)
            print("Transformed Grasp Locations : \n", transformed_grasp_locations)

            # display the transformed grasp locations on the scene image
            for i, location in enumerate(transformed_grasp_locations):
                cv.circle(scene_image, (int(location[0]), int(location[1])), 5, (0, 255, 0), -1)
                # cv.putText(scene_image, "H" + str(i), (int(location[0])+2, int(location[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # get the mid point of the transformed grasp locations
            mid_point = get_midpoint(transformed_grasp_locations)

            # draw the mid point on the scene image
            cv.circle(scene_image, (int(mid_point[0]), int(mid_point[1])), 5, (0, 0, 255), -1)
            # cv.putText(scene_image, "Mid Point", (int(mid_point[0]), int(mid_point[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # display the scene image
            cv.imshow('Scene Image - Mid Point of grasp locations', scene_image)
            cv.waitKey(0)

        except:
            print("Error in finding the transformed grasp locations")
            print("Calculating the centroid of the object...")
            
            # use centroid of the scene image as the grasp location
            centroid = get_centroid(scene_image)
            print("Centroid of the template image : ", centroid)

            # draw the centroid on the scene image
            cv.circle(scene_image, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
            cv.putText(scene_image, "C", (int(centroid[0])+2, int(centroid[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # display the scene image
            cv.imshow('Scene Image - Centroid of the 2D object', scene_image)
            cv.waitKey(0)

            return

    else:

        print("Not enough matching keypoints to calculate homography matrix.")
        print("Calculating the centroid of the object...")

        # use centroid of the scene image as the grasp location
        centroid = get_centroid(scene_image)
        print("Centroid of the template image : ", centroid)

        # draw the centroid on the scene image
        cv.circle(scene_image, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
        cv.putText(scene_image, "C", (int(centroid[0])+2, int(centroid[1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # display the scene image
        cv.imshow('Scene Image - Centroid of the 2D object', scene_image)
        cv.waitKey(0)


if __name__ == '__main__':

    # print the opencv version
    print("OpenCV version : ", cv.__version__)

    # print the numpy version
    print("Numpy version : ", np.__version__)

    # execute the main fucntion     
    main(input("Enter the object class : ").strip())
    
    # destroy all the windows
    cv.destroyAllWindows()
