import torch
import config as cfg
from model import SSD
from dec_loss import DEC_loss
import torch.optim as optim
from dataset_pascal import PASCALVOC
import transforms
from viewDatasets import viewDatasets_DEC
import os
import numpy as np
import cv2
import time
import dec_eval
import pickle
import torch.nn as nn
import shutil


from torch.optim import lr_scheduler


import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

layers = {
    'resnet50':  [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3]
}


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs,0),targets

class DEC_Module(object):
    def __init__(self, multigpu, resume):
        self.model = SSD(num_classes=cfg.num_classes,
                                    num_blocks=cfg.mbox,
                                    top_k=cfg.top_k,
                                    conf_thresh=cfg.conf_thresh,
                                    nms_thresh=cfg.nms_thresh,
                                    variance=cfg.variance)
        if resume is not None:
            print('Resuming training weights from {} ...'.format(resume))
            resume_dict = torch.load(resume)
            resume_dict_update = {}
            for k in resume_dict:
                if k.startswith('module') and not k.startswith('module_list'):
                    resume_dict_update[k[7:]] = resume_dict[k]
                else:
                    resume_dict_update[k] = resume_dict[k]
            self.model.load_state_dict(resume_dict_update)
        else:
            resnet = "resnet101"
            print('Resuming weights from {} ...'.format(resnet))
            pre_trained_dict = model_zoo.load_url(model_urls[resnet])
            model_dict = self.model.state_dict()
            updated_dict = {k: v for k,v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(updated_dict)
            self.model.load_state_dict(model_dict)

        self.multigpu = multigpu


    def train(self, vis=False):
        print("begin training....")

        if not os.path.exists('weights'):
            os.mkdir('weights')

        # Device settings
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        if self.multigpu:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)


        eval_model = SSD(num_classes=cfg.num_classes,
                                    num_blocks=cfg.mbox,
                                    top_k=cfg.top_k,
                                    conf_thresh=cfg.conf_thresh,
                                    nms_thresh=cfg.nms_thresh,
                                    variance=cfg.variance)
        eval_model = eval_model.to(device1)


        for item in self.model.parameters():
            print(item.requires_grad)

        total_epoch = cfg.epoch

        criterion = DEC_loss(num_classes=cfg.num_classes, variances=cfg.variance, device=device)

        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=cfg.init_lr,
                              momentum=0.9,
                              weight_decay=cfg.weight_decay)

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_epoch, gamma=0.1)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)

        print('Loading Datasets...')
        dsets = PASCALVOC(root=cfg.root,
                          image_sets=cfg.train_sets,
                          transform=transforms.DEC_transforms(phase='train',
                                                              size=cfg.img_size,
                                                              mean=cfg.means,
                                                              std=cfg.std))

        dsets_val = PASCALVOC(root=cfg.root,
                              image_sets=cfg.test_sets,
                              transform=transforms.DEC_transforms(phase='val',
                                                                  size=cfg.img_size,
                                                                  mean=cfg.means,
                                                                  std=cfg.std))

        dset_loaders = torch.utils.data.DataLoader(dsets,
                                                   cfg.batch_size,
                                                   num_workers=4,
                                                   shuffle=True,
                                                   collate_fn=detection_collate,
                                                   pin_memory=True)
        if vis:
            viewDatasets_DEC(dset_loaders)

        train_loss_dict = []
        mAP_dict = []
        for epoch in range(total_epoch):
            print('Epoch {}/{}'.format(epoch, total_epoch - 1))
            print('-' * 10)
            for phase in ['train','val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                    running_loss = 0.0
                    for data in dset_loaders:
                        inputs, target = data
                        inputs = inputs.to(device)
                        target = [item.to(device) for item in target]

                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs, phase)
                            # backprop
                            loss_l, loss_c = criterion(outputs, target)
                            loss = loss_l + loss_c

                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item()

                    epoch_loss = running_loss / len(dsets)
                    print('{} Loss: {:.6}'.format(epoch, epoch_loss))

                    train_loss_dict.append(epoch_loss)
                    np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
                    if epoch % 5 == 0:
                        torch.save(self.model.state_dict(),
                                   os.path.join('weights', '{:d}_{:.4f}_model.pth'.format(epoch, epoch_loss)))
                    torch.save(self.model.state_dict(), os.path.join('weights', 'end_model.pth'))

                else:
                    if epoch%5==0:
                        model_dict = self.model.state_dict()
                        val_dict = {k[7:]: v for k, v in model_dict.items()}
                        eval_model.load_state_dict(val_dict)
                        maps = self.eval(device1, eval_model, dsets_val)
                        mAP_dict.append(maps)
                        np.savetxt('mAP.txt', mAP_dict, fmt='%.6f')


    def test(self):
        print('testing, evaluation mode...')
        self.model.eval()

        print('loading data...')
        dsets = PASCALVOC(root=cfg.root,
                          image_sets=cfg.test_sets,
                          transform=transforms.DEC_transforms(phase='val',
                                                              size=cfg.img_size,
                                                              mean=cfg.means,
                                                              std=cfg.std))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        num_imgs = len(dsets)
        test_timer = Timer()
        cv2.namedWindow('img')
        for i in range(num_imgs):
            print('testing {}...'.format(dsets.img_ids[i]))
            img, target = dsets.__getitem__(i)
            ori_img = cv2.imread(dsets._imgpath % dsets.img_ids[i])
            h, w, c = ori_img.shape

            x = img.unsqueeze(0)
            x = x.to(device)

            test_timer.tic()
            detections = self.model(x, 'test')
            detect_time = test_timer.toc(average=False)
            print('test time: {}'.format(detect_time))
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:,0].gt(0.).expand(5,dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1,5)
                if dets.shape[0]==0:
                    continue
                if j:
                    boxes = dets[:,1:]
                    boxes[:,0] *= h
                    boxes[:,1] *= w
                    boxes[:,2] *= h
                    boxes[:,3] *= w
                    scores = dets[:,0].cpu().numpy()
                    for box, score in zip(boxes,scores):
                        y1,x1,y2,x2 = box
                        y1 = int(y1)
                        x1 = int(x1)
                        y2 = int(y2)
                        x2 = int(x2)

                        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                        cv2.putText(ori_img, cfg.VOC_CLASSES[int(j)]+"%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 0, 255))
            cv2.imshow('img', ori_img)
            k = cv2.waitKey(0)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        cv2.destroyAllWindows()
        exit()

    def eval(self, device, eval_model, dsets):
        # print('evaluation mode...')
        # self.model.eval()
        eval_model.eval()
        output_dir = cfg.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        num_imgs = len(dsets)
        total_time = 0

        det_file = os.path.join(output_dir, 'detections.pkl')
        # print('Detecting bounding boxes...')
        all_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}

        for i in range(num_imgs):
            img, target = dsets.__getitem__(i)
            ori_img = cv2.imread(dsets._imgpath % dsets.img_ids[i])
            h, w, c = ori_img.shape

            x = img.unsqueeze(0)
            x = x.to(device)

            _t['im_detect'].tic()
            detections = eval_model(x, 'test')
            detect_time = _t['im_detect'].toc(average=False)

            # ignore the background boxes
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.shape[0] == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= h
                boxes[:, 1] *= w
                boxes[:, 2] *= h
                boxes[:, 3] *= w
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            # print('img-detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_imgs, detect_time))
            total_time += detect_time

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            f.close()

        print('Saving the results...')
        for cls_ind, cls in enumerate(cfg.labelmap):
            # print('Writing {:s} VOC results file'.format(cls))
            filename = dec_eval.get_voc_results_file_template('test', cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dsets.img_ids):
                    dets = all_boxes[cls_ind + 1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

        # print('Evaluating detections....')
        print('average time is {}'.format(float(total_time)/num_imgs))
        maps = dec_eval.do_python_eval(output_dir=output_dir, use_07=True)
        return maps

    def eval_single(self):
        print('evaluation mode...')
        self.model.eval()

        print('loading data...')
        dsets = PASCALVOC(root=cfg.root,
                          image_sets=cfg.test_sets,
                          transform=transforms.DEC_transforms(phase='val',
                                                              size=cfg.img_size,
                                                              mean=cfg.means,
                                                              std=cfg.std))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        output_dir = cfg.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)

        num_imgs = len(dsets)

        det_file = os.path.join(output_dir, 'detections.pkl')
        print('Detecting bounding boxes...')
        all_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        total_time = 0
        for i in range(num_imgs):
            img, target = dsets.__getitem__(i)
            ori_img = cv2.imread(dsets._imgpath % dsets.img_ids[i])
            h, w, c = ori_img.shape

            x = img.unsqueeze(0)
            x = x.to(device)

            _t['im_detect'].tic()
            detections = self.model(x, 'test')
            detect_time = _t['im_detect'].toc(average=False)
            total_time += detect_time
            # ignore the background boxes
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.shape[0] == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= h
                boxes[:, 1] *= w
                boxes[:, 2] *= h
                boxes[:, 3] *= w
                boxes[:,0] = np.maximum(0., boxes[:,0])
                boxes[:,1] = np.maximum(0., boxes[:,1])
                boxes[:,2] = np.minimum(h, boxes[:,2])
                boxes[:,3] = np.minimum(w, boxes[:,3])
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                all_boxes[j][i] = cls_dets

            print('img-detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_imgs, detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print('average time is {}'.format(float(total_time)/num_imgs))
        print('Saving the results...')
        for cls_ind, cls in enumerate(cfg.labelmap):
            print('Writing {:s} VOC results file'.format(cls))
            filename = dec_eval.get_voc_results_file_template('test', cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dsets.img_ids):
                    dets = all_boxes[cls_ind + 1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

        print('Evaluating detections....')

        dec_eval.do_python_eval(output_dir=output_dir, use_07=True)


