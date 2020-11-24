num_classes = 21
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
mbox = [4, 6, 6, 6, 6, 4, 4]
variance = [0.1, 0.2]
feature_maps = [65, 33, 17, 9, 5, 3, 1]

min_sizes = [  20.52,   51.3,   133.38,  215.46,  297.54,  379.62,  461.7 ]
max_sizes = [  51.3,   133.38,  215.46,  297.54,  379.62,  461.7,   543.78]

steps = [8, 16, 31, 57, 103, 171, 513]
top_k = 200

# detect settings
conf_thresh =  0.01
nms_thresh = 0.45

# Training settings
img_size = 321
batch_size = 2
workers = 0
epoch = 100
MULTI_SCALE_TRAIN = False
init_lr = 1e-4
end_lr = 1e-6
momentum = 0.9
weight_decay = 0.0005
warmup_epoch = 2
# lr_decay_epoch = 50
milestones = [120, 170, 220]

# data directory
root = 'E:\YOLOV4\VOCdevkit'

# train_sets = [('2007','train'),('2007','val'),('2012', 'train'),('2012', 'val')]
train_sets = [('2007','train'),('2007','val')]
test_sets = [('2007', 'test')]

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

VOC_CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# dec evaluation
output_dir = 'output'

labelmap = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
