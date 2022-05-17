from io_helper import *

def main():
    
    # get the template image from the directory
    template_image = cv.imread('template/allen_key.jpg', 0)

    # get the grasping location from the user as mouse clicked points
    grasp_locations = get_grasp_locations(template_image)

if __name__ == '__main__':
    main()