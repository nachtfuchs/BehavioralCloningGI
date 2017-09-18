import numpy as np
import cv2

def resize_image(input_image):
    '''@brief Reduce image size for computational reasons.
    '''
    x_size = 64
    y_size = 32
    return cv2.resize(input_image, (x_size, y_size))

def crop_image(input_image):
    '''@brief Cut off the sky and lower part of the picture since it contains
              little information about the track.
    '''
    offset_low = 40
    offset_high = 140
    return input_image[offset_low:offset_high]

def change_color_space(input_image):
    ''' Return image in S dimension of HSV color space
    '''
    return  cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)[:, :, 1]

def image_wrapper(input_image):
    '''@brief summarize all relevant image processing functions
    '''
    return resize_image(change_color_space(input_image)).reshape(1, 32, 64, 1)

def preprocess(img): 
    ''' @brief Looks like this is not necessary anymore
    '''
    image = (cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1],
                          (32, 16))).reshape(1, 16, 32, 1)
    return image