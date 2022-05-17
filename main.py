from io_helper import *

def main():
    
    # get the template image from the directory
    template_image = cv.imread('template/allen_key.jpg', cv.IMREAD_ANYCOLOR)

    # get the grasping location from the user as mouse clicked points
    grasp_locations = get_grasp_locations(template_image)

    # print the mouse clicked points
    print(grasp_locations)




if __name__ == '__main__':
    main()