import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
import config as cfg

class ImageFolder_COCO(data.Dataset):
    def __init__(self, root, dataType, transform=None):
        self.root = root
        self.coco = 0
        annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
        self.coco = COCO(annFile)

        self.image_set_index = sorted(self.coco.getImgIds())
        for image_id in self.image_set_index:
            anns = self.coco.imgToAnns[image_id]
            if len(anns)==0:
                self.image_set_index.remove(image_id)

        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))


        self.transform = transform
        self._imgInd_to_coco_imgId = dict(zip(range(len(self.image_set_index)),self.image_set_index))
        self._imgpath = os.path.join(root, dataType, '%s')
        self.classToInd = dict(zip(self.classes, range(len(self.classes))))


    def load_gt(self, anns, height, width):
        res = []
        for obj in range(len(anns)):
            cat_id = anns[obj]['category_id']
            img_label_name= str(self.coco.cats[cat_id]['name'])
            bb = anns[obj]['bbox'] # x1, y1, w, h
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            x1 = float(x1)/width
            x2 = float(x2)/width
            y1 = float(y1)/height
            y2 = float(y2)/height

            if abs(y2-y1)<=1e-15 or abs(x2-x1)<=1e-15:
                continue

            res.append([y1,x1,y2,x2,self.classToInd[img_label_name]])
        return res


    def __getitem__(self, index):
        image_id = self._imgInd_to_coco_imgId[index]
        img = cv2.imread(self._imgpath % self.coco.imgs[image_id]['file_name'])
        anns = self.coco.imgToAnns[image_id]
        height, width, channels = img.shape
        target = self.load_gt(anns, height, width)

        if len(anns)==0:
            target = None
        else:
            if self.transform is not None:
                target = np.array(target)
                img, bboxes, labels = self.transform(img, target[:,:4], target[:,4])
                target = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.transpose((2,0,1)).copy())

            if isinstance(target, np.ndarray):
                target = torch.FloatTensor(target)

        return img, target

    def __len__(self):
        return len(self._imgInd_to_coco_imgId)



class ImageFolder_COCO_eval(data.Dataset):
    def __init__(self, root, dataType, transform=None):
        self.root = root
        self.coco = 0
        annFile = '{}/annotations/instances_{}.json'.format(root, dataType)
        self.coco = COCO(annFile)

        self.image_set_index = sorted(self.coco.getImgIds())
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))

        self.transform = transform
        self._imgInd_to_coco_imgId = dict(zip(range(len(self.image_set_index)),self.image_set_index))
        self._imgpath = os.path.join(root, dataType, '%s')
        self.classToInd = dict(zip(self.classes, range(len(self.classes))))

    def __getitem__(self, index):
        image_id = self._imgInd_to_coco_imgId[index]
        img = cv2.imread(self._imgpath % self.coco.imgs[image_id]['file_name'])
        img = img.astype(np.float32)
        img = cv2.resize(img, (cfg.img_size, cfg.img_size))
        img = img / 255
        means = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        means = np.array(means, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img[:, :, 0] -= means[0]
        img[:, :, 1] -= means[1]
        img[:, :, 2] -= means[2]
        img[:, :, 0] /= std[0]
        img[:, :, 1] /= std[1]
        img[:, :, 2] /= std[2]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose((2,0,1)).copy())


        return img

    def __len__(self):
        return len(self._imgInd_to_coco_imgId)