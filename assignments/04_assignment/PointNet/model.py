from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, segmentation = False, d=1024):
        super(PointNetfeat, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the feature extractor. ##
        ## ------------------------------------------- ##
        self.d=d
        self.segmentation=segmentation
        self.mlp1=nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=64,kernel_size=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.mlp2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU())
        self.mlp3=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=d,kernel_size=1),
            nn.BatchNorm1d(num_features=d),
            nn.ReLU())
        
        
    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        B=x.size()[0]
        N=x.size()[1]
        x=x.transpose(1,2)
        x=self.mlp1(x)
        local_feature=x
        x=self.mlp2(x)
        x=self.mlp3(x)
        x=x.transpose(1,2)
        perpoint_feature=x
        global_feature,_=torch.max(input=x,dim=1)
        if self.segmentation==True:
            global_feature=global_feature.reshape(B,self.d,1).repeat((1,1,N))
            return torch.cat([global_feature,local_feature],dim=1)
        else:
            return global_feature,perpoint_feature


class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.k=k
        self.feat=PointNetfeat(segmentation=False,d=1024)
        self.mlp1=nn.Sequential(
            nn.Linear(in_features=1024,out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        self.mlp2=nn.Sequential(
            nn.Linear(in_features=512,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
        self.mlp3=nn.Linear(in_features=256,out_features=k)
        
        self.log_softmax=nn.LogSoftmax(dim=1)


    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x,vis_feature=self.feat.forward(x)
        x=self.mlp1(x)
        x=self.mlp2(x)
        x=self.mlp3(x)
        x=self.log_softmax(x)
        return x,vis_feature
        


class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.k=k
        self.feat=PointNetfeat(segmentation=False,d=256)
        self.mlp1=nn.Sequential(
            nn.Linear(in_features=256,out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.mlp2=nn.Linear(in_features=128,out_features=k)
        self.log_softmax=nn.LogSoftmax(dim=1)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x,vis_feature=self.feat.forward(x)
        x=self.mlp1(x)
        x=self.mlp2(x)
        x=self.log_softmax(x)
        return x,vis_feature


class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the segmentation head. ##
        ## ------------------------------------------- ##
        self.k=k
        self.feat=PointNetfeat(segmentation=True,d=1024)
        self.mlp1=nn.Sequential(
            nn.Conv1d(in_channels=1088,out_channels=512,kernel_size=1),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512,out_channels=256,kernel_size=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.mlp2=nn.Linear(in_features=128,out_features=k)
        self.log_softmax=nn.LogSoftmax(dim=2)

    def forward(self, x):
        '''
            Input:
                x: the concatenated global feature and local feature. # (B, d+64, N)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x=self.feat.forward(x)
        x=self.mlp1(x)
        x=x.transpose(1,2)
        x=self.mlp2(x)
        x=self.log_softmax(x)
        return x


