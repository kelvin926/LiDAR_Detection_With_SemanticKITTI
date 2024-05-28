import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_ops.pointnet2_modules as pointnet2

class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet2Classification, self).__init__()
        self.num_classes = num_classes

        self.sa1 = pointnet2.PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=3,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = pointnet2.PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=320,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = pointnet2.PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=640,
            mlp=[256, 512, 1024],
            group_all=True
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
