import cv2
import config as cfg
import numpy as np
import random

def ViewDatasets(dset_loaders):
    cv2.namedWindow('img')
    for epoch in range(10):
        for data in dset_loaders:
            img_batch, target_batch = data
            img_batch = img_batch.numpy()
            for idx in range(cfg.batch_size):
                img = img_batch[idx].transpose(1, 2, 0)
                img[:,:,0] *= cfg.std[0]
                img[:,:,1] *= cfg.std[1]
                img[:,:,2] *= cfg.std[2]
                img[:,:,0] += cfg.means[0]
                img[:,:,1] += cfg.means[1]
                img[:,:,2] += cfg.means[2]
                img = img*255
                img = np.uint8(img)
                ori_img = img.copy()
                bboxes = target_batch[idx].numpy()[:,:-1]
                labels = np.asarray(target_batch[idx].numpy()[:,-1])
                height, width, _ = img.shape
                img = ori_img.copy()
                for idxi in range(len(bboxes)):
                    box = bboxes[idxi]
                    label = labels[idxi]
                    y1, x1, y2, x2 = box
                    x1 = np.int32(x1 * width)
                    y1 = np.int32(y1 * height)
                    x2 = np.int32(x2 * width)
                    y2 = np.int32(y2 * height)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                    cv2.putText(img, cfg.VOC_CLASSES[int(label)], (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255))
                cv2.imshow('img', img)
                k = cv2.waitKey(0)
                if k & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
    cv2.destroyAllWindows()
    exit()


def viewDatasets_SEG(dsets):
    for i in range(100):
        inputs, gt_boxes, gt_classes, gt_masks = dsets.__getitem__(i)
        img = inputs.numpy().transpose(1, 2, 0)

        img[:, :, 0] = img[:, :, 0] + 104
        img[:, :, 1] = img[:, :, 1] + 117
        img[:, :, 2] = img[:, :, 2] + 123

        gt_boxes = gt_boxes.numpy()
        gt_classes = gt_classes.numpy()
        gt_masks = gt_masks.numpy()
        num_obj, _ = gt_boxes.shape

        if num_obj:
            for i in range(num_obj):
                y11, x11, y22, x22 = gt_boxes[i, :]*cfg.img_size
                y11 = np.maximum(int(y11), 0)
                y22 = np.minimum(int(y22), cfg.img_size )
                x11 = np.maximum(int(x11), 0)
                x22 = np.minimum(int(x22), cfg.img_size )
                cur_gt_mask = gt_masks[i, :, :]
                cur_inst_cls = gt_classes[i]
                [r, c] = np.where(cur_gt_mask == 1)
                y1 = np.min(r)
                y2 = np.max(r)
                x1 = np.min(c)
                x2 = np.max(c)
                mask = np.zeros(cur_gt_mask.shape, dtype=np.float32)
                mask[cur_gt_mask == 1] = 1.
                color = (random.random(), random.random(), random.random())
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                mskd = img * mask
                clmsk = np.ones(mask.shape) * mask
                clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
                clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
                clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
                img = img + 0.8 * clmsk - 0.8 * mskd
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                cv2.rectangle(img, (x11, y11), (x22, y22), (0, 255, 255), 2, 2)
                cv2.putText(img, cfg.VOC_CLASSES[int(cur_inst_cls)], (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255))

                cv2.namedWindow('img')
                cv2.imshow('img', np.uint8(img))
                k = cv2.waitKey(0)
                if k == 27:
                    cv2.destroyAllWindows()
                    exit(1)
                cv2.destroyWindow('img')