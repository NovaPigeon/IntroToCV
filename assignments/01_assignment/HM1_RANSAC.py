import numpy as np
from utils import draw_save_plane_with_points, normalize
import math


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")
    
    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     
    
    # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    sample_time = math.floor(math.log(0.001,1-(100/130)**3))+1 
    distance_threshold = 0.05

    # sample points group

    points_sampled=np.random.randint(low=0,high=noise_points.shape[0],size=(sample_time,3))
    # estimate the plane with sampled points group
    point_0=noise_points[points_sampled[:,0]]
    point_1=noise_points[points_sampled[:,1]]
    point_2=noise_points[points_sampled[:,2]]

    vector_0=point_0-point_1
    vector_1=point_0-point_2

    normal_vector = np.cross(vector_0, vector_1)
    
    As=normal_vector[:,0].reshape((normal_vector.shape[0],1))
    Bs=normal_vector[:,1].reshape((normal_vector.shape[0],1))
    Cs=normal_vector[:,2].reshape((normal_vector.shape[0],1))
    Ds=-(As*(point_0[:,0].reshape((normal_vector.shape[0],1)))
       +Bs*(point_0[:,1].reshape((normal_vector.shape[0],1)))
       +Cs*(point_0[:,2].reshape((normal_vector.shape[0],1))))
    
    # evaluate inliers (with point-to-plance distance < distance_threshold)
    distances= np.abs(
               np.tile(As,(1,noise_points.shape[0]))*noise_points[:,0]+
               np.tile(Bs,(1,noise_points.shape[0]))*noise_points[:,1]+
               np.tile(Cs,(1,noise_points.shape[0]))*noise_points[:,2]+
               np.tile(Ds,(1,noise_points.shape[0]))
               )/ \
               np.sqrt(
                np.tile(As,(1,noise_points.shape[0]))*np.tile(As,(1,noise_points.shape[0]))+
                np.tile(Bs,(1,noise_points.shape[0]))*np.tile(Bs,(1,noise_points.shape[0]))+
                np.tile(Cs,(1,noise_points.shape[0]))*np.tile(Cs,(1,noise_points.shape[0]))
               )
    
    inliers=distances<distance_threshold
    inliers_cnt=np.sum(inliers,axis=1)
    plane_chosen_idx=np.argmax(inliers_cnt)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    chosen_inliers=noise_points[np.where(distances[plane_chosen_idx]<distance_threshold)]
    center=np.mean(chosen_inliers,axis=0)
    centered_points=chosen_inliers-center
    U,D,V_T=np.linalg.svd(centered_points)
    A,B,C=V_T[-1]
    D=-np.dot(V_T[-1],center)
    pf=[A,B,C,D]
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

