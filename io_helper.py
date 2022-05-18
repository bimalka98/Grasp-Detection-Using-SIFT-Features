import cv2 as cv
from cv2 import morphologyEx
import numpy as np


NumberOfGraspPoints = 2 # number of grasp points per object
global CurrentGraspPoint
CurrentGraspPoint = 0

# mouse callback function to draw the circle when the mouse is clicked
def draw_circle(event,x,y,flags,params):

    """
    draw a circle when the mouse is clicked.
    """

    global CurrentGraspPoint
    p = params[0]
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(params[1],(x,y),2,(255,0,0),-1)
        p.append((x,y))
        CurrentGraspPoint += 1

# fucntion to get the grasp locations from the user
def get_grasp_locations(template_image):

    """
    get the grasping points of the object using the mouse clicked points.
    """

    # list to keep the mouse clicked points
    grasp_locations = []
    
    # initialize the variable to its default value
    global CurrentGraspPoint
    CurrentGraspPoint = 0

    # create a window to show the template image
    cv.namedWindow('Template Image', cv.WINDOW_NORMAL)

    # make it full screen
    cv.setWindowProperty('Template Image', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # arrange the params for the mouse callback function
    params = [grasp_locations, template_image]

    # set the mouse callback function
    cv.setMouseCallback('Template Image', draw_circle, params)

    # get mouse clicked points    
    while(1):
        cv.imshow('Template Image', template_image)
        if CurrentGraspPoint == NumberOfGraspPoints:            
            # wait for 0.5 seconds
            cv.waitKey(500)

            # break the loop
            break

        if cv.waitKey(20) & 0xFF == 27:            
            # break the loop
            break
    
    # return the list of mouse clicked points
    return grasp_locations

#fucntion ot get the centrid of an object
def get_centroid(scene_image):

    """
    get the centroid of the object.
    """

    # convert the image to grayscale
    template = cv.cvtColor(scene_image.copy(), cv.COLOR_BGR2GRAY)

    # gausian blur the image
    template = cv.GaussianBlur(template, (5, 5), 0)

    # otsu's thresholding
    ret, thresh = cv.threshold(template, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # declaring morphological operation element
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # morphological closing
    closed_image = cv.morphologyEx(thresh, cv.MORPH_CLOSE, element)

    # find contours
    contours, hierarchy = cv.findContours(closed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # get the largest contour
    max_contour = max(contours, key=cv.contourArea)

    # get the centroid of the largest contour
    M = cv.moments(max_contour)

    # get the centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # return the centroid
    return (cx, cy)

    
def get_midpoint(transformed_grasp_locations):

    """
    get the midpoint of two points.
    """
    p1=transformed_grasp_locations[0]
    p2=transformed_grasp_locations[1]

    # get the midpoint
    midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    # return the midpoint
    return midpoint    
    

    