import numpy as np
from numpy import random
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels


class ConvertFromInts(object):
    def __call__(self, img, bboxes=None, labels=None):
        return img.astype(np.float32), bboxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, img, bboxes=None, labels=None):
        h,w,c = img.shape
        bboxes = np.asarray(bboxes)
        bboxes[:,0::2] *= h
        bboxes[:,1::2] *= w
        return img, bboxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, bboxes, labels

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img, bboxes=None, labels=None):
        if self.current=='BGR' and self.transform=='HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current=='HSV' and self.transform=='BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, bboxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            img[:,:,1] *= random.uniform(self.lower, self.upper)
        return img, bboxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            img[:,:,0] += random.uniform(-self.delta, self.delta)
            img[:,:,0][img[:,:,0]>360.0]-=360.0
            img[:,:,0][img[:,:,0]<0.0]+=360.0
        return img, bboxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, bboxes, labels

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, bboxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [RandomContrast(),
                   ConvertColor(current='BGR', transform='HSV'),
                   RandomSaturation(),
                   RandomHue(),
                   ConvertColor(current='HSV', transform='BGR'),
                   RandomContrast()]
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, bboxes=None, labels=None):
        img, bboxes, labels = self.rb(img, bboxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        img, bboxes, labels = distort(img, bboxes, labels)
        img, bboxes, labels = self.rln(img, bboxes, labels)
        return img, bboxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            return img, bboxes, labels
        h,w,c = img.shape
        ratio = random.uniform(1,4)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
        expand_img[:,:,:] = self.mean
        expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
        img = expand_img

        bboxes[:,0::2] += float(int(y1))
        bboxes[:,1::2] += float(int(x1))

        return img, bboxes, labels

def intersect(boxes_a, box_b):
    max_yx = np.minimum(boxes_a[:,2:], box_b[2:])
    min_yx = np.maximum(boxes_a[:,:2], box_b[:2])
    inter = np.clip((max_yx-min_yx), a_min=0., a_max=np.inf)
    return inter[:,0]*inter[:,1]

def jaccard_numpy(boxes_a, box_b):
    # boxes_a: float
    # box_b: int
    inter = intersect(boxes_a, box_b)
    area_a = ((boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1]))
    area_b = ((box_b[2]-box_b[0])*(box_b[3]-box_b[1]))
    union = area_a+area_b-inter
    return inter/union #float



class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, img, bboxes=None, labels=None):
        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, bboxes, labels
            min_iou, max_iou = mode

            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_img = img
                w = random.uniform(0.3*width, width)
                h = random.uniform(0.3*height, height)
                if h/w<0.5 or h/w>2:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                overlap = jaccard_numpy(bboxes, rect)
                if overlap.min()<min_iou and max_iou<overlap.max():
                    continue
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                centers = (bboxes[:,:2]+bboxes[:,2:])/2.0
                mask1 = (rect[0]<centers[:,0])*(rect[1]<centers[:,1])
                mask2 = (rect[2]>centers[:,0])*(rect[3]>centers[:,1])
                mask = mask1*mask2
                if not mask.any():
                    continue
                current_boxes = bboxes[mask,:].copy()
                current_labels = labels[mask]
                current_boxes[:,:2] = np.maximum(current_boxes[:,:2], rect[:2])
                current_boxes[:,:2]-=rect[:2]
                current_boxes[:,2:] = np.minimum(current_boxes[:,2:], rect[2:])
                current_boxes[:,2:]-=rect[:2]
                return current_img, current_boxes, current_labels

class RandomMirror(object):
    def __call__(self, img, bboxes, classes):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1]
            bboxes[:,1::2] = w-bboxes[:,3::-2]
        return img, bboxes, classes


class ToPercentCoords(object):
    def __call__(self, img, bboxes=None, labels=None):
        h,w,c = img.shape
        bboxes[:, 0]/=h
        bboxes[:, 1]/=w
        bboxes[:, 2]/=h
        bboxes[:, 3]/=w
        return img, bboxes, labels

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size))  # PIL will normalize to [-1,1] or [0,1]?
        return img, bboxes, labels

class SubtractMeans(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img, bboxes=None, labels=None):
        img = img.astype(np.float32)
        img = img/255
        img[:,:,0] -= self.mean[0]
        img[:,:,1] -= self.mean[1]
        img[:,:,2] -= self.mean[2]
        img[:,:,0] /= self.std[0]
        img[:,:,1] /= self.std[1]
        img[:,:,2] /= self.std[2]
        return img, bboxes, labels



class DEC_transforms(object):
    def __init__(self, phase, size, mean, std):
        if phase == 'train':
            self.augment = Compose(transforms=[ConvertFromInts(),
                                               ToAbsoluteCoords(),
                                               PhotometricDistort(),
                                               Expand(mean),
                                               RandomSampleCrop(),
                                               RandomMirror(),
                                               ToPercentCoords(),
                                               Resize(size),
                                               SubtractMeans(mean, std)])
        else:
            self.augment = Compose(transforms=[ConvertFromInts(),
                                               Resize(size),
                                               SubtractMeans(mean, std)])
    def __call__(self, img, bboxes=None, labels=None):
        return self.augment(img, bboxes, labels)
