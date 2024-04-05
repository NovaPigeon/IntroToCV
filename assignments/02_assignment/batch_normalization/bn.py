import numpy as np
import cv2
import os


# eps may help you to deal with numerical problem
eps = 1e-5
def bn_forward_test(x, gamma, beta, mean, var):

    #----------------TODO------------------
    # Implement forward 
    x_hat=(x-mean)/np.sqrt(var+eps)
    out=x_hat*gamma+beta
    #----------------TODO------------------

    return out

def bn_forward_train(x, gamma, beta):

    #----------------TODO------------------
    # Implement forward
    sample_mean=np.mean(x,axis=0)
    sample_var=np.var(x,axis=0)
    x_hat=(x-sample_mean)/np.sqrt(sample_var+eps)
    out=x_hat*gamma+beta
    
    global rms_mean
    global rms_var
    global rho
    rms_mean=rho*rms_mean+(1-rho)*sample_mean
    rms_var=rho*rms_var+(1-rho)*sample_var
    #----------------TODO------------------

    # save intermidiate variables for computing the gradient when backward
    cache = (gamma, x, sample_mean, sample_var, x_hat)
    
    return out, cache
    
def bn_backward(dout, cache):

    #----------------TODO------------------
    # Implement backward
    # mean=sum(x)/N
    # var=sum((x-mean)**2)/N
    # x_hat=x-mean/(var+eps)**0.5
    # out=gamma*x_hat+beta
    gamma, x, sample_mean, sample_var, x_hat = cache
    N,_=dout.shape
    dgamma=np.sum(x_hat*dout,axis=0)
    dbeta=np.sum(dout,axis=0)
    dx_hat=dout*gamma
    dvar=np.sum(dx_hat*(x-sample_mean)*(-0.5)*(sample_var+eps)**(-1.5),axis=0)
    dmean=np.sum(dx_hat*(-(sample_var+eps)**(-0.5)),axis=0)+dvar*np.sum(-2*(x-sample_mean)/N,axis=0)
    dx=dx_hat*(sample_var+eps)**(-0.5)+dvar*2*(x-sample_mean)/N+dmean/N
    #----------------TODO------------------

    return dx, dgamma, dbeta

# This function may help you to check your code
def print_info(x):
    print('mean:', np.mean(x,axis=0))
    print('var:',np.var(x,axis=0))
    print('------------------')
    return 

if __name__ == "__main__":
    HW_dir = os.path.dirname(os.path.dirname(__file__))

    # input data
    train_data = np.zeros((9,784)) 
    for i in range(9):
        train_data[i,:] = cv2.imread(os.path.join(HW_dir, "mnist_subset", f"{i}.png"), cv2.IMREAD_GRAYSCALE).reshape(-1)/255.
    gt_y = np.zeros((9,1)) 
    gt_y[0] =1  

    val_data = np.zeros((1,784)) 
    val_data[0,:] = cv2.imread(os.path.join(HW_dir, "mnist_subset", "9.png"), cv2.IMREAD_GRAYSCALE).reshape(-1)/255.
    val_gt = np.zeros((1,1)) 

    np.random.seed(14)

    # Intialize MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784,16)
    MLP_layer_2 = np.random.randn(16,1)

    # Initialize gamma and beta
    gamma = np.random.randn(16)
    beta = np.random.randn(16)

    lr=1e-1
    loss_list=[]

    # ---------------- TODO -------------------
    # compute mean and var for testing
    # add codes anywhere as you need
    # ---------------- TODO -------------------
    rms_mean = 0.
    rms_var = 0.
    rho=0.5

    # training 
    for i in range(50):
        # Forward
        output_layer_1 = train_data.dot(MLP_layer_1)
        output_layer_1_bn, cache = bn_forward_train(output_layer_1, gamma, beta)
        output_layer_1_act = 1 / (1+np.exp(-output_layer_1_bn))  #sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1+np.exp(-output_layer_2))  #sigmoid activation function

        # compute loss 
        loss = -( gt_y * np.log(pred_y) + (1-gt_y) * np.log(1-pred_y)).sum()
        print("iteration: %d, loss: %f" % (i+1 ,loss))
        loss_list.append(loss)

        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)
        grad_pred_y = -(gt_y/pred_y) + (1-gt_y)/(1-pred_y)
        grad_activation_func = grad_pred_y * pred_y * (1-pred_y) 
        grad_layer_2 = output_layer_1_act.T.dot(grad_activation_func)
        grad_output_layer_1_act = grad_activation_func.dot(MLP_layer_2.T)
        grad_output_layer_1_bn  = grad_output_layer_1_act * (1-output_layer_1_act) * output_layer_1_act
        grad_output_layer_1, grad_gamma, grad_beta = bn_backward(grad_output_layer_1_bn, cache)
        grad_layer_1 = train_data.T.dot(grad_output_layer_1)

        # update parameters
        gamma -= lr * grad_gamma
        beta -= lr * grad_beta
        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2
    
    # validate
    output_layer_1 = val_data.dot(MLP_layer_1)
    output_layer_1_bn = bn_forward_test(output_layer_1, gamma, beta, rms_mean, rms_var)
    
    output_layer_1_act = 1 / (1+np.exp(-output_layer_1_bn))  #sigmoid activation function
    output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
    pred_y = 1 / (1+np.exp(-output_layer_2))  #sigmoid activation function
    loss = -( val_gt * np.log(pred_y) + (1-val_gt) * np.log(1-pred_y)).sum()
    print("validation loss: %f" % (loss))
    loss_list.append(loss)

    os.makedirs(os.path.join(os.path.join(HW_dir), "results"), exist_ok=True)
    np.savetxt(os.path.join(os.path.join(HW_dir), "results", "bn_loss.txt"), loss_list)