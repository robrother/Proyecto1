#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline
from scipy.ndimage.filters import gaussian_filter
import scipy
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import os
os.listdir("/home/cidetec/Documents/")


x1_left_li = []
x2_left_li = []
x1_right_li = []
x2_right_li = []
x1_left_a = 0
x2_left_a = 0
x1_right_a = 0
x2_right_a = 0
avg_qty = 5
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
hough_thres = 15 # minimum number of votes (intersections in Hough grid cell)
min_line_len = 55 #minimum number of pixels making up a line
max_line_gap = 5 # maximum gap in pixels between connectable line segments

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, image, alfa=0.8, beta=1., gama=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alfa+ img * beta + gama
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(image, alfa, img, beta, gama)

def mvg_avg_left(x1, x2, qty = 5):
    """
    Calculates a simple moving average
    `x1` is the x1 value
    `x2` is the x2 value
    `qty` is the number of values to consider in the moving average
    """
    global x1_left_li, x2_left_li, x1_left_a, x2_left_a
    if len(x1_left_li) > qty:
        x1_left_li.pop(0) 
        x2_left_li.pop(0)
    x1_left_li.append(x1)
    x2_left_li.append(x2)
    x1_left_a = int(sum(x1_left_li) / float(len(x1_left_li)))
    x2_left_a = int(sum(x2_left_li) / float(len(x2_left_li)))
    
    
def mvg_avg_right(x1, x2, qty = 5):
    """
    Calculates a simple moving average
    `x1` is the x1 value
    `x2` is the x2 value
    `qty` is the number of values to consider in the moving average
    """
    global x1_right_li, x2_right_li, x1_right_a, x2_right_a
    if len(x1_right_li) > qty: 
        x1_right_li.pop(0) 
        x2_right_li.pop(0) 
    x1_right_li.append(x1)
    x2_right_li.append(x2)
    x1_right_a = int(sum(x1_right_li) / float(len(x1_right_li)))
    x2_right_a = int(sum(x2_right_li) / float(len(x2_right_li)))

def draw_straight_lines(img, lines, y_min = 315, color=[0, 127, 255], thickness=16):
    # Generate lists to store values
# Generate lists to store values
    x1_left = []
    x2_left = []
    y1_left = []
    y2_left = []
    x1_right = []
    x2_right = []
    y1_right = []
    y2_right = []
    
    # Add the line values to the appropriate list
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1) # get the slope
            if slope >= 0:
                # positive slope, right line
                x1_right.append(x1)
                x2_right.append(x2)
                y1_right.append(y1)
                y2_right.append(y2)
            else:
                # negative slope, left line
                x1_left.append(x1)
                x2_left.append(x2)
                y1_left.append(y1)
                y2_left.append(y2)
            
    # Compute the average value for each point
    x1_left_avg = (sum(x1_left) / float(len(x1_left))) if len(x1_left) != 0 else 0
    x2_left_avg = (sum(x2_left) / float(len(x2_left))) if len(x2_left) != 0 else 0
    y1_left_avg = (sum(y1_left) / float(len(y1_left))) if len(y1_left) != 0 else 0
    y2_left_avg = (sum(y2_left) / float(len(y2_left))) if len(y2_left) != 0 else 0
    x1_right_avg = (sum(x1_right) / float(len(x1_right))) if len(x1_right) != 0 else 0
    x2_right_avg = (sum(x2_right) / float(len(x2_right))) if len(x2_right) != 0 else 0
    y1_right_avg = (sum(y1_right) / float(len(y1_right))) if len(y1_right) != 0 else 0
    y2_right_avg = (sum(y2_right) / float(len(y2_right))) if len(y2_right) != 0 else 0
    
    # Get the slope denominator for each side
    left_denom = (x2_left_avg - x1_left_avg)
    right_denom = (x2_right_avg - x1_right_avg) 
    
    # Get the x,y dimension of the image
    x = img.shape[1]
    y = img.shape[0]
    
    # Calculate the extended lines and get the moving average
    if left_denom != 0:
        # Get the slope
        left_slope = ((y2_left_avg - y1_left_avg) / left_denom)
        
        # Get the intercept
        left_b = y2_left_avg - left_slope * x2_left_avg
        
        # Calculate lines
        x1 = int((y_min - left_b) / left_slope)
        x2 = int((y - left_b) / left_slope)
        
        # Update moving averages if difference < 20% from average
        mvg_avg_left(x1, x2, avg_qty)
        
    if right_denom != 0:
        # Get the slope
        right_slope = ((y2_right_avg - y1_right_avg) / right_denom)
        
        # Get the intercept
        right_b = y2_right_avg - right_slope * x2_right_avg
        
        # Calculate lines
        x1 = int((y_min - right_b) / right_slope)
        x2 = int((y - right_b) / right_slope)
        
        # Update moving averages
        mvg_avg_right(x1, x2, avg_qty)
    
    # Plot the lines
    cv2.line(img, (x1_left_a, y_min), (x2_left_a, y), color, thickness)
    cv2.line(img, (x1_right_a, y_min), (x2_right_a, y), color, thickness)



def get_slope(x1,y1,x2,y2):
    return ((y2-y1)/(x2-x1))

def get_x(x,y,dy,slope):
    return
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print(lines)
    # Make a RGB shape of the correct dimensions
    shape = (img.shape[0], img.shape[1], 3)
    line_img = np.zeros(shape, dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img

def hough_straight_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with straight hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_straight_lines(line_img, lines)
    return line_img

def process_image(image):
 # Leemos la imagen
 #image = mpimg.imread('/home/cidetec/Documents/exit_ramp.jpg',0)
 #La imagen original la transformamos a escala de grises
 gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
 #convertimos la imagen a HLS
 img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
 #plt.imshow(img_hls)
 #plt.show()
 #Escogemos un rango objetivo de los valores de amarillo que buscamos en la imagen
 lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
 upper_yellow = np.array([30, 255, 255], dtype= 'uint8')
 #Escogemos un rango objetivo de los valores de blanco que buscamos en la imagen
 lower_white = np.array([0, 200, 0], dtype = 'uint8')
 upper_white = np.array([180, 255, 255], dtype= 'uint8')
 #aplicamos un enmascaramiento con esos valores
 mask_yellow = cv2.inRange(img_hls, lower_yellow, upper_yellow)
 mask_white =  cv2.inRange(img_hls, lower_white, upper_white)
 mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
 mask_yw_image = cv2.bitwise_and(gray, mask_yw)
 #plt.imshow(mask_yw_image)
 #plt.show()
 # Aplicamos un filtro gausiano para reducir el ruido en la imagen
 borroso = gaussian_filter(mask_yw_image, sigma=1)
 #plt.imshow(borroso)
 #plt.show()
 # Ahora aplicamos deteccion de Canny para bordes
 edges = cv2.Canny(borroso,100,200)
 #plt.imshow(edges)
 #plt.show()
 #Nos enfocamos solo en la region de interes
 mask = np.zeros_like(edges)                                     #Return an empty array with shape and type of input  
 ignore_mask_color = 255 
 imshape = image.shape
 lower_left = [imshape[1]/9,imshape[0]]
 lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
 top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
 top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
 vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
 cv2.fillPoly(mask, vertices, ignore_mask_color)                 #cv.FillPoly(img, polys, color, lineType=8, shift=0) 
 edges2 = cv2.bitwise_and(edges, mask)   
 masked_image = edges2
 lines = hough_straight_lines(edges2, rho, theta, hough_thres, \
                                  min_line_len, max_line_gap) # Hough transform to lines
 weighted = weighted_img(lines, image) # add the lines to the initial image 

 def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
     """
     `img` should be the output of a Canny transform.
         
     Returns an image with hough lines drawn.
     """
     lines = cv2.HoughLinesP(edges2, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
     draw_lines(line_img, lines)
     return line_img

 result = weighted

 return result


white_output = '/home/cidetec/Documents/solidYellowLeftErick.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("/home/cidetec/Documents/solidYellowLeft.mp4").subclip(0,5)
clip1 = VideoFileClip("/home/cidetec/Documents/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
