import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad=np.sqrt(np.power(x_grad,2)+np.power(y_grad,2))
    direction_grad=np.arctan2(y_grad,x_grad)
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   

    grad_dir=grad_dir*180/np.pi
    def nms_single_direction(
                             dx,
                             dy,
                             alpha1,
                             alpha2,
                             alpha3,
                             alpha4):
        if (alpha3==157.5) & (alpha4==-157.5):
            mag_i,mag_j=np.where(((alpha1 < grad_dir) & (grad_dir <= alpha2))
                        |((alpha3 < grad_dir) | (grad_dir <= -alpha4)))
        else:
            mag_i,mag_j=np.where(((alpha1 < grad_dir) & (grad_dir <= alpha2))
                            |((alpha3 < grad_dir) & (grad_dir <= -alpha4)))
        
        valid_indices=(mag_i>=abs(dx)) &\
                      (mag_i<grad_mag.shape[0]-abs(dx))&\
                      (mag_j>=abs(dy)) &\
                      (mag_j<grad_mag.shape[1]-abs(dy))
        mag_i=mag_i[valid_indices]
        mag_j=mag_j[valid_indices]
        mid_points=grad_mag[mag_i,mag_j]
        parallel_points=grad_mag[mag_i-dx,mag_j-dy]
        antiparallel_points=grad_mag[mag_i+dx,mag_j+dy]

        edge=np.where((mid_points>parallel_points) & (mid_points>antiparallel_points))
        edge_i=mag_i[edge]
        edge_j=mag_j[edge]

        return (edge_i,edge_j)

    dir1_edge_i,dir1_edge_j=nms_single_direction(1,0,67.5,112.5,-112.5,67.5)
    dir2_edge_i,dir2_edge_j=nms_single_direction(-1,1,112.5,157.5,-67.5,-22.5)
    dir3_edge_i,dir3_edge_j=nms_single_direction(0,-1,-22.5,22.5,157.5,-157.5)
    dir4_edge_i,dir4_edge_j=nms_single_direction(-1,-1,22.5,67.5,-157.5,-112.5)

    NMS_output=np.zeros_like(grad_mag)
    NMS_output[dir1_edge_i,dir1_edge_j]=grad_mag[dir1_edge_i,dir1_edge_j]
    NMS_output[dir2_edge_i,dir2_edge_j]=grad_mag[dir2_edge_i,dir2_edge_j]
    NMS_output[dir3_edge_i,dir3_edge_j]=grad_mag[dir3_edge_i,dir3_edge_j]
    NMS_output[dir4_edge_i,dir4_edge_j]=grad_mag[dir4_edge_i,dir4_edge_j]
    
    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.10
    high_ratio = 0.30
    x,y=np.where(img>=high_ratio)
    q=list(zip(list(x),list(y)))
    
    output=np.zeros_like(img)
    output[x,y]=1

    dx=[0,0,-1,-1,-1,1,1,1]
    dy=[1,-1,-1,0,1,-1,0,1]

    while q:
        p=q.pop(0)
        for i in range(8):
            new_x=p[0]+dx[i]
            new_y=p[1]+dy[i]
            if new_x <0 or new_x>=img.shape[0]:
                continue
            if new_y<0 or new_y>=img.shape[1]:
                continue
            if output[new_x][new_y]==1:
                continue
            if img[new_x][new_y]>=low_ratio:
                q.append((new_x,new_y))
                output[new_x][new_y]=1
        

    return output



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
