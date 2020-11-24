import torch.utils.data as data
import os
import sys
import cv2
import numpy as np
import torch
from utils.data_augment import *

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

class load_gt(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, height, width):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text)==1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['ymin', 'xmin', 'ymax', 'xmax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)-1
                cur_pt = float(cur_pt)/height if i%2==0 else float(cur_pt)/width
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]
        return res


class PASCALVOC(data.Dataset):
    def __init__(self, root, image_sets, phase, mean, std, img_size=513):
        self.img_size = img_size  # For Multi-training
        self.root = root
        self.image_sets = image_sets
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.img_ids = []
        self.load_gt = load_gt()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC'+year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.img_ids.append((rootpath, line.strip()))
        if phase == 'train':
            print("DEC_transforms size {}".format(self.img_size))
            self.augment = Compose(transforms=[ConvertFromInts(),
                                               ToAbsoluteCoords(),
                                               PhotometricDistort(),
                                               Expand(mean),
                                               RandomSampleCrop(),
                                               RandomMirror(),
                                               ToPercentCoords(),
                                               Resize(self.img_size),
                                               SubtractMeans(mean, std)])
        else:
            self.augment = Compose(transforms=[ConvertFromInts(),
                                               Resize(self.img_size),
                                               SubtractMeans(mean, std)])

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        target = self.load_gt(target, height, width)
        target = np.array(target)
        img, bboxes, labels = self.transform(img, target[:,:4], target[:,4])
        target = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose((2,0,1)).copy())

        if isinstance(target, np.ndarray):
            target = torch.FloatTensor(target)

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def transform(self,img, bboxes, labels):
        img, bboxes, labels = self.augment(img, bboxes, labels)
        # img = cv2.resize(img, (self.img_size, self.img_size))
        # print("resize {}".format(img.shape))
        # img = img.astype(np.float32)
        # img = img / 255
        # img[:, :, 0] -= self.mean[0]
        # img[:, :, 1] -= self.mean[1]
        # img[:, :, 2] -= self.mean[2]
        # img[:, :, 0] /= self.std[0]
        # img[:, :, 1] /= self.std[1]
        # img[:, :, 2] /= self.std[2]
        return img, bboxes, labels
