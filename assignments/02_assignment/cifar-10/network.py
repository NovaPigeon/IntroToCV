import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        # ----------TODO------------
        # define a network 
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=72,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=72),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # ----------TODO------------
        self.fc=nn.Sequential(
            nn.Linear(72*4*4,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self, x):

        # ----------TODO------------
        # network forwarding 
        x=self.conv(x)
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        # ----------TODO------------
        return x


if __name__ == '__main__':
    import torch
    from torch.utils.tensorboard  import SummaryWriter
    from dataset import CIFAR10
    writer = SummaryWriter(log_dir='../experiments/network_structure')
    net = ConvNet()
    train_dataset = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, num_workers=2)
    # Write a CNN graph. 
    # Please save a figure/screenshot to '../results' for submission.
    
    for imgs, labels in train_loader:
        writer.add_graph(net, imgs)
        writer.close()
        break 
