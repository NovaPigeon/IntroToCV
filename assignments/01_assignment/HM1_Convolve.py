import numpy as np
from utils import read_img, write_img

def padding(img:np.ndarray, padding_size:int, type:str):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    w,h=img.shape
    padding_img:np.ndarray=np.zeros((w+2*padding_size,h+2*padding_size))
    if type=="zeroPadding":
        padding_img[padding_size:w+padding_size,padding_size:h+padding_size]=img
        return padding_img
    elif type=="replicatePadding":
        padding_img[padding_size:w+padding_size,padding_size:h+padding_size]=img
        padding_img[0:padding_size,0:padding_size]=img[0][0]
        padding_img[0:padding_size,h+padding_size:h+2*padding_size]=img[0][h-1]
        padding_img[w+padding_size:w+2*padding_size,0:padding_size]=img[w-1][0]
        padding_img[w+padding_size:w+2*padding_size,h+padding_size:h+2*padding_size]=img[w-1][h-1]

        padding_img[0:padding_size,padding_size:h+padding_size]=img[0,0:h]
        padding_img[padding_size:w+padding_size,0:padding_size]=img[0:w,0].reshape(w,1)
        padding_img[w+padding_size:w+2*padding_size,padding_size:h+padding_size]=img[w-1,0:h]
        padding_img[padding_size:w+padding_size,h+padding_size:h+2*padding_size]=img[0:w,h-1].reshape(w,1)
        return padding_img


def convol_with_Toeplitz_matrix(img:np.ndarray, kernel:np.ndarray):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img:np.ndarray = padding(img,1,"zeroPadding").reshape(64,1)

    #build the Toeplitz matrix and compute convolution
    kernel=kernel.reshape(9)
    k1=list(kernel[0:3])
    k2=list(kernel[3:6])
    k3=list(kernel[6:9])
    l1=k1+[0]*5+k2+[0]*5+k3+[0]*46 # 3+5+3+5+3+46=65=64+1(循环往右移一位)
    l2=l1*6+[0]*2 # 补足 2 位，分块
    l3=l2*5+l1*5+k1+[0]*5+k2+[0]*5+k3 # 处理最后一行
    topl_mat=np.array(l3).reshape(36,64)
    output=(topl_mat@padding_img).reshape(6,6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    
    image_height, image_width = img.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    output = np.zeros((output_height, output_width))

    kernel_i,kernel_j=np.meshgrid(np.arange(kernel_height),np.arange(kernel_width),indexing="ij")
    kernel_i=kernel_i.reshape(1,kernel_height*kernel_width)
    kernel_j=kernel_j.reshape(1,kernel_height*kernel_width)
    kernel_i-=kernel_height//2
    kernel_j-=kernel_width//2
    img_i,img_j=np.meshgrid(np.arange(image_height)[kernel_height//2:image_height-kernel_height//2],
                            np.arange(image_width)[kernel_width//2:image_width-kernel_width//2],indexing="ij")
    
    img_i=img_i.reshape((output_width*output_height,1))
    img_i=np.repeat(img_i,kernel_height*kernel_width).\
        reshape((output_width*output_height,kernel_height*kernel_width))
    
    img_j=img_j.reshape((output_width*output_height,1))
    img_j=np.repeat(img_j,kernel_height*kernel_width).\
        reshape((output_width*output_height,kernel_height*kernel_width))
    
    img_i+=kernel_i
    img_j+=kernel_j

    
    img_sliding_window=img[img_i,img_j]
    kernel=kernel.reshape(1,kernel_height*kernel_width)
    kernel=np.repeat(kernel,output_width*output_height,axis=0).\
        reshape((output_width*output_height,kernel_height*kernel_width))
    
    output=np.sum(img_sliding_window*kernel,axis=1).reshape((output_height,output_width))
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    