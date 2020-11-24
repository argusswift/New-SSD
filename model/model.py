import torch
import math
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from .priorbox import PriorBox
import torch.nn.functional as F

layer_cfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'extra': [512, 'S', 1024, 256, 'S', 512, 128, 'S', 256, 128, 256],
    'pred': [512, 1024, 2048, 1024, 512, 256, 256]
}

#############################################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

#############################################################################################
# build extra
def add_extras(inplances, batch_norm=False):
    layers = []
    in_channels = 2048
    flag = False
    for k, v in enumerate(inplances):
        if in_channels != 'S':  # skip M when necessary
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, inplances[k+1], 3, stride=2, padding=1)
                out_channels = inplances[k + 1]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], stride=1)
                out_channels = v

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            flag = not flag
        in_channels = v
    return layers
#############################################################################################
# build conf locs
def build_conf(cfg_i, num_blocks, num_classes):
    layers = []
    for i in range(len(num_blocks)):
        layers += [torch.nn.Conv2d(cfg_i[i], num_blocks[i]*num_classes, kernel_size=3,padding=1)]
    return layers

def build_locs(cfg_i, num_blocks):
    layers = []
    for i in range(len(num_blocks)):
        layers += [torch.nn.Conv2d(cfg_i[i], num_blocks[i]*4, kernel_size=3,padding=1)]
    return layers

#############################################################################################
## nms, decode, detect
def decode(locs, priors, variances):
    # locs:    num_priors x 4
    # priors:  num_priors x 4
    boxes = torch.cat([priors[:, :2] + locs[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(locs[:, 2:] * variances[1])], 1)
    boxes[:, 0] = boxes[:, 0]-boxes[:, 2]/2
    boxes[:, 1] = boxes[:, 1]-boxes[:, 3]/2
    boxes[:, 2] = boxes[:, 2]+boxes[:, 0]
    boxes[:, 3] = boxes[:, 3]+boxes[:, 1]
    #boxes[:, :2] -= boxes[:, 2:] / 2
    #boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, nms_thresh=0.5, top_k=200):
    # boxes shape[-1, 4]
    # scores shape [-1,]
    scores = scores
    boxes = boxes
    keep = scores.new(scores.size(0)).zero_().long()# create a new tensor
    if boxes.numel() == 0: # return the total elements number in boxes
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(y2-y1,x2-x1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    yy1 = boxes.new()  # create a new tensor of the same type
    xx1 = boxes.new()
    yy2 = boxes.new()
    xx2 = boxes.new()
    h = boxes.new()
    w = boxes.new()
    count = 0
    while idx.numel()>0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # doing about remains...
        # select the remaining boxes
        torch.index_select(y1, dim=0, index=idx, out=yy1)
        torch.index_select(x1, dim=0, index=idx, out=xx1)
        torch.index_select(y2, dim=0, index=idx, out=yy2)
        torch.index_select(x2, dim=0, index=idx, out=xx2)

        # calculate the inter boxes clamp with box i
        yy1 = torch.clamp(yy1, min=y1[i])
        xx1 = torch.clamp(xx1, min=x1[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        xx2 = torch.clamp(xx2, max=x2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2-xx1
        h = yy2-yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w*h

        rem_areas = torch.index_select(area, dim=0, index=idx)
        union = (rem_areas-inter)+area[i]
        IoU = inter/union
        idx = idx[IoU.le(nms_thresh)]
    return keep, count


class Detect(object):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh, variance):
        self.num_classes = num_classes
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.output = torch.zeros(1, self.num_classes, self.top_k, 5)

    def __call__(self, locs, confs, priors):
        # locs:   batch x num_priors x 4
        # confs:  num_priors x 21
        # priors: num_priors x 4 [cy,cx,h,w]
        num_batch = locs.size(0)
        num_priors = priors.size(0)
        self.output.zero_()
        # p_confs: 21 x num_priors
        if num_batch == 1:
            p_confs = confs.t().contiguous().unsqueeze(0)
        else:
            p_confs = confs.view(num_batch, num_priors, self.num_classes).transpose(2, 1)
            self.output.expand_(num_batch, self.num_classes, self.top_k, 5)

        locs = locs.data.cpu()
        priors = priors.data.cpu()
        p_confs = p_confs.data.cpu()

        # Decoding...
        for i in range(num_batch):
            decoded_boxes_i = decode(locs[i], priors, torch.Tensor(self.variance))
            p_conf_i = p_confs[i].clone()
            for cl in range(1, self.num_classes):
                cl_mask = p_conf_i[cl].gt(self.conf_thresh)
                p_conf_i_cl = p_conf_i[cl][cl_mask]
                if p_conf_i_cl.dim() == 0:
                    continue
                loc_mask = cl_mask.unsqueeze(1).expand_as(decoded_boxes_i)
                p_boxes_i_cl = decoded_boxes_i[loc_mask].view(-1,4)
                if p_boxes_i_cl.shape[0]<1:
                    continue
                ids, count = nms(boxes=p_boxes_i_cl,
                                 scores=p_conf_i_cl,
                                 nms_thresh=self.nms_thresh)
                self.output[i, cl, :count] = torch.cat((p_conf_i_cl[ids[:count]].unsqueeze(1),
                                                        p_boxes_i_cl[ids[:count]]),1)

        return self.output
#############################################################################################
def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)

class Attention(nn.Module):
    def __init__(self, c):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        f = self.conv1(x)   # [bs,c',h,w]
        g = self.conv2(x)   # [bs,c',h,w]
        h = self.conv3(x)   # [bs,c',h,w]

        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)    # [bs,N,c']
        g = hw_flattern(g)              # [bs,c',N]
        h = hw_flattern(h)              # [bs,c,N]
        h = torch.transpose(h, 1, 2)    # [bs,N,c]

        s = torch.matmul(f,g)           # [bs,N,N]
        beta = F.softmax(s, dim=-1)

        o = torch.matmul(beta,h)        # [bs,N,c]
        o = torch.transpose(o, 1, 2)
        o = o.view(x.shape)
        x = o + x
        return x

#512, 1024, 2048, 1024, 512, 256, 256
def make_attention():
    layers = []
    layers.append(Attention(512))
    layers.append(Attention(1024))
    layers.append(Attention(2048))
    layers.append(Attention(1024))
    layers.append(Attention(512))
    layers.append(Attention(256))
    layers.append(Attention(256))
    return layers
# #############################################################################################
def fusionModule():
    conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
    conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    conv3 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
    layers = [conv1, conv2, conv3]
    return layers
#############################################################################################
class SSD(nn.Module):
    def __init__(self, num_classes, num_blocks,
                 top_k, conf_thresh, nms_thresh,
                 variance,use_attention=False):
        super(SSD,self).__init__()
        self.att = use_attention
        self.num_classes = num_classes
        ############################################################################################
        self.inplanes = 64
        layers = [3, 4, 23, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64,  layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        #self.L2Norm = L2Norm(n_channels=512, scale=20)
        self.extra_layers = nn.ModuleList(add_extras(layer_cfg['extra'], batch_norm=True))
        self.conf_layers = nn.ModuleList(build_conf(layer_cfg['pred'], num_blocks, num_classes))
        self.locs_layers = nn.ModuleList(build_locs(layer_cfg['pred'], num_blocks))
        self.prior_boxes = PriorBox()
        self.prior_boxes = self.prior_boxes.forward()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prior_boxes = self.prior_boxes.to(device)
        if self.att == True:
            self.fusion_layers = nn.ModuleList(fusionModule())
            self.fusion_bn = nn.BatchNorm2d(768) #256*3
            self.fusion_conv = nn.Conv2d(768, 512, kernel_size=1)
            self.att_layers = nn.ModuleList(make_attention())

        self.softmax = nn.Softmax(dim=1)
        self.detect = Detect(num_classes=num_classes,
                             top_k=top_k,
                             conf_thresh=conf_thresh,
                             nms_thresh=nms_thresh,
                             variance=variance)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, phase=None):
        feat = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)              #(2L, 256L, 129L, 129L)
        x = self.layer2(x)              #(2L, 512L, 65L, 65L)
        feat += [x]
        feat0 = x
        x = self.layer3(x)              #(2L, 1024L, 33L, 33L)
        feat += [x]
        feat1 = x
        x = self.layer4(x)
        feat += [x]
        feat2 = x
        for k, v in enumerate(self.extra_layers):
            x = v(x)
            if k in [5, 11, 17, 23]:
                feat += [x]
        if self.att == True:
            ######### fusion #################################################
            feat0 = self.fusion_layers[0](feat0)
            feat1 = F.upsample_bilinear(self.fusion_layers[1](feat1),size=(65,65))
            feat2 = F.upsample_bilinear(self.fusion_layers[2](feat2),size=(65,65))
            feat[0] = F.relu(self.fusion_conv(self.fusion_bn(torch.cat([feat0, feat1, feat2], dim=1))))
            #################################################################
            feat_new = []
            for (x,l) in zip(feat, self.att_layers):
                feat_new.append(l(x))
        ########## PreEnd #################################################
        locs = []
        conf = []
        for (x, l, c) in zip(feat, self.locs_layers, self.conf_layers):
            locs += [l(x).permute(0,2,3,1).contiguous()]
            conf += [c(x).permute(0,2,3,1).contiguous()]

        locs = torch.cat([o.view(o.size(0),-1) for o in locs], dim=1)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf], dim=1)

        if phase == 'test':
            output = self.detect(locs.view(locs.size(0), -1, 4),
                                 self.softmax(conf.view(-1, self.num_classes)),
                                 self.prior_boxes.type(type(x.data)))
        else:
            output = (locs.view(locs.size(0),-1,4),
                      conf.view(conf.size(0),-1,self.num_classes),
                      self.prior_boxes)
        return output