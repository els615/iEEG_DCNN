import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

save_dir = '/mmfs1/data/schwarex/ieeg/neuralNetworks/identity/resnet_noSoftmax/kdef_features'

#
device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
#
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
#
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1503, zero_init_residual=False, # change 100 to 7
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, #could change channels to 1 
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = x.clone().detach() 
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x.clone().detach() 
        x = self.layer2(x)
        x3 = x.clone().detach() 
        x = self.layer3(x)
        x4 = x.clone().detach() 
        x = self.layer4(x)
        x5 = x.clone().detach() 
        x = self.avgpool(x)
        x6 = x.clone().detach() 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #
        #x = F.log_softmax(x)
        return x, x1, x2, x3, x4, x5, x6
    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

import os
import glob
from random import *
import csv
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_pickle('/mmfs1/data/schwarex/neuralNetworks/datasets/KDEF_5views/kdef_5views.pkl')

#
# Implement the data loader.
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms, utils
from PIL import Image
#
class KDEFDataset(Dataset):
    """KDEF dataset."""
    def __init__(self, t_set, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    # take only the part of the file that's relevant for the current usage
        self.dataAll = t_set
        #self.dataManyImages = (dataInfo_temp[dataInfo_temp['numImages']>1]).reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.dataAll)
    def __getitem__(self, idx):
        img_name =  self.dataAll.iloc[idx,3]     
        image = Image.open(img_name)
        label = self.dataAll.iloc[idx,0] #identity label
        index = self.dataAll.iloc[idx,5] #index labels to pair image with extracted features
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop(388),
            transforms.Resize(48), 
        #    transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
                                            ])
        image = data_transforms(image)
        sample = {'image': image, 'label': label, 'index':index}
        if self.transform:
            sample = self.transform(sample)
        return sample
#
#
for rep in range(1,2): 
    checkpoint_file = '/mmfs1/data/schwarex/ieeg/neuralNetworks/identity/resnet_noSoftmax/199_%d.ckpt' % rep # each checkpoint file really should be num -1 
    net = resnet18(pretrained=False, progress=True)
    savedModel=torch.load(checkpoint_file)
    state_dict=savedModel['state_dict']
    optimizer=savedModel['optimizer']
    epoch=savedModel['epoch']
    net.load_state_dict(state_dict)
    net.to(device)
    for name, param in net.named_parameters():
        param.requires_grad = False
    # Create loader
    KDEF_test_transformed = KDEFDataset(t_set=df,
                                            transform=None
                                            )
    testloader = DataLoader(KDEF_test_transformed, batch_size=64,
                            shuffle=True, num_workers=2,pin_memory=True)
    net.eval()
    score = []
    image_pred_final = None
    image_pred_layer1 = None
    image_pred_layer2 = None
    image_pred_layer3 = None
    image_pred_layer4 = None
    image_pred_layer5 = None
    image_pred_layer6 = None
    idx_arr = None
    count = 0
    for i, data in enumerate(testloader):
        # get the inputs
        images = data['image']
        labels = data['label']
        indices = data['index']
        tmp = []
        tmp = torch.squeeze(labels.long())
        tmpInd = []
        tmpInd = torch.squeeze(indices.long())
        images, labels, indices = images.to(device=device, dtype=torch.float), tmp.to(device=device, dtype=torch.float), tmpInd.to(device=device, dtype=torch.float)
        # forward + backward + optimize
        all_outputs = net(images)
        output_final = all_outputs[0].cpu().data.detach().numpy()
        output_layer1 = all_outputs[1].cpu().data.detach().numpy()
        output_layer2 = all_outputs[2].cpu().data.detach().numpy()
        output_layer3 = all_outputs[3].cpu().data.detach().numpy()
        output_layer4 = all_outputs[4].cpu().data.detach().numpy()
        output_layer5 = all_outputs[5].cpu().data.detach().numpy()
        output_layer6 = all_outputs[6].cpu().data.detach().numpy()
        output_argmax = np.argmax(output_final)
        labels_numpy = labels.cpu().data.numpy()
        indices = indices.cpu().data.numpy()
        score = np.concatenate((score,(labels_numpy==output_argmax).astype(int)),axis=0)
        try:
            image_pred_final = np.concatenate((image_pred_final, output_final))
            image_pred_layer1 = np.concatenate((image_pred_layer1, output_layer1))
            image_pred_layer2 = np.concatenate((image_pred_layer2, output_layer2))
            image_pred_layer3 = np.concatenate((image_pred_layer3, output_layer3))
            image_pred_layer4 = np.concatenate((image_pred_layer4, output_layer4))
            image_pred_layer5 = np.concatenate((image_pred_layer5, output_layer5))
            image_pred_layer6 = np.concatenate((image_pred_layer6, output_layer6))
            idx_arr = np.concatenate((idx_arr, indices))
        except ValueError:
            image_pred_final = output_final
            image_pred_layer1 = output_layer1
            image_pred_layer2 = output_layer2
            image_pred_layer3 = output_layer3
            image_pred_layer4 = output_layer4
            image_pred_layer5 = output_layer5
            image_pred_layer6 = output_layer6
            idx_arr = indices
    list_of_tuples = list(zip(image_pred_final, image_pred_layer1, image_pred_layer2, image_pred_layer3, image_pred_layer4, image_pred_layer5, image_pred_layer6, idx_arr)) 
    df2 = pd.DataFrame(list_of_tuples, columns = ['final_output', 'conv1', 'lay1', 'lay2', 'lay3', 'lay4', 'avg_pool', 'index'])
    df2.to_pickle(save_dir+'/kdef_features_%d.pkl' % rep) 
    # merge
    new_df2 = pd.merge(df,df2,on=['index'])
    new_df2.to_pickle(save_dir+'/kdef_features_merged_%d.pkl' % rep)

