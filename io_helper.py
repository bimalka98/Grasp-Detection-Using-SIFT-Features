import cv2 as cv
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
            # wait for 2 seconds
            cv.waitKey(1000)

            # break the loop
            break

        if cv.waitKey(20) & 0xFF == 27:            
            # break the loop
            break
    
    # return the list of mouse clicked points
    return grasp_locations



    