import utils.gpu as gpu
from model.ssd_loss import SSD_loss
import torch.optim as optim
from utils.dataset_pascal import PASCALVOC
from utils.viewDatasets import ViewDatasets
import tqdm
import pickle
import argparse
from eval.evaluator import *
from utils.tools import *
import config.ssd_config as cfg
from utils import cosine_lr_scheduler
from model.model import SSD
import torch.utils.model_zoo as model_zoo
from eval.dec_eval import get_voc_results_file_template, do_python_eval


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs,0),targets


class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id, vis, mode=None):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.epoch = 100
        self.weight_path = weight_path
        self.resume = resume
        self.mode = mode
        self.multi_scale_train = cfg.MULTI_SCALE_TRAIN
        print('Loading Datasets...')
        self.train_dataset = PASCALVOC(img_size=cfg.img_size,root=cfg.root,
                          image_sets=cfg.train_sets,
                          phase='trainval',mean=cfg.means, std=cfg.std)
        self.val_dataset = PASCALVOC(img_size=cfg.img_size,root=cfg.root,
                              image_sets=cfg.test_sets,
                              phase='test',mean=cfg.means, std=cfg.std)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.workers,
                                           collate_fn=detection_collate,
                                           shuffle=True)
        if vis:
            ViewDatasets(self.train_dataloader)
        self.SSD = SSD(num_classes=cfg.num_classes,
                                    num_blocks=cfg.mbox,
                                    top_k=cfg.top_k,
                                    conf_thresh=cfg.conf_thresh,
                                    nms_thresh=cfg.nms_thresh,
                                    variance=cfg.variance).to(self.device)
        self.optimizer = optim.SGD(self.SSD.parameters(), lr=cfg.init_lr,
                                   momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        self.criterion = SSD_loss(num_classes=cfg.num_classes, variances=cfg.variance, device=self.device)

        self.__load_model_weights(self.weight_path, self.resume, self.mode)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.init_lr,
                                                          lr_min=cfg.end_lr,
                                                          warmup=cfg.warmup_epoch*len(self.train_dataloader))


    def __load_model_weights(self, weight_path, resume, mode):
        model_urls = {
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        }
        if resume or mode:
            print('Resuming training weights from {} ...'.format(weight_path))
            resume_dict = torch.load(weight_path)
            resume_dict_update = {}
            for k in resume_dict:
                if k.startswith('module') and not k.startswith('module_list'):
                    resume_dict_update[k[7:]] = resume_dict[k]
                else:
                    resume_dict_update[k] = resume_dict[k]
            self.SSD.load_state_dict(resume_dict_update)
        else:
            resnet = "resnet101"
            print('Resuming weights from {} ...'.format(resnet))
            pre_trained_dict = model_zoo.load_url(model_urls[resnet])
            weight_path = 'weight/resnet101-5d3b4d8f.pth'
            pre_trained_dict = torch.load(weight_path)
            model_dict = self.SSD.state_dict()
            updated_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(updated_dict)
            self.SSD.load_state_dict(model_dict)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.SSD.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        print(self.SSD)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        mAP_dict = []
        for epoch in range(self.start_epoch, self.epochs):
            mloss = torch.zeros(3)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.SSD.train()
                    i = 0
                    for data in self.train_dataloader:
                        i += 1
                        self.scheduler.step(len(self.train_dataloader) * epoch + i)
                        inputs, target = data
                        inputs = inputs.to(self.device)
                        target = [item.to(self.device) for item in target]

                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.SSD(inputs, phase)
                            # backprop
                            loss_l, loss_c = self.criterion(outputs, target)
                            loss = loss_l + loss_c

                            loss.backward()
                            self.optimizer.step()

                        # Update running mean of tracked metrics
                        loss_items = torch.tensor(
                            [loss_l, loss_c, loss]
                        )
                        mloss = (mloss * i + loss_items) / (i + 1)
                        if i % 10 == 0:
                            print("=== Epoch:[{:3}/{}],step:[{:3}/{}],img_size:[{:3}],total_loss:{:.4f}|loss_l:{:.4f}|loss_c:{:.4f}|lr:{:.10f}".format(
                                    epoch,
                                    self.epochs,
                                    i,
                                    len(self.train_dataloader) - 1,
                                    self.train_dataset.img_size,
                                    mloss[2],
                                    mloss[0],
                                    mloss[1],
                                    self.optimizer.param_groups[0]["lr"],
                                )
                            )
                        # multi-sclae training (320-608 pixels) every 10 batches
                        if self.multi_scale_train and (i+1)%10 == 0:
                            self.train_dataset.img_size = (random.choice(range(10, 20)) * 32)
                            print("multi_scale_img_size : {}".format(self.train_dataset.img_size))
                else:
                    if epoch>50:
                        maps = self.eval(self.device, self.SSD, self.val_dataset)
                        mAP_dict.append(maps)
                        np.savetxt('mAP.txt', mAP_dict, fmt='%.6f')
                        self.__save_model_weights(epoch, maps)

    def test(self):
                print('testing mode...')
                self.SSD.eval()

                print('loading data...')
                dsets = PASCALVOC(img_size=cfg.img_size,
                                  root=cfg.root,
                                  image_sets=cfg.test_sets,
                                  phase='test',mean=cfg.means,std=cfg.std)
                num_imgs = len(dsets)
                test_timer = Timer()
                for i in range(num_imgs):
                    print('testing {}...'.format(dsets.img_ids[i]))
                    img, target = dsets.__getitem__(i)
                    ori_img = cv2.imread(dsets._imgpath % dsets.img_ids[i])
                    h, w, c = ori_img.shape

                    x = img.unsqueeze(0)
                    x = x.to(self.device)

                    test_timer.tic()
                    detections = self.SSD(x, 'test')
                    detect_time = test_timer.toc(average=False)
                    print('test time: {}'.format(detect_time))
                    for j in range(1, detections.size(1)):
                        dets = detections[0, j, :]
                        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                        dets = torch.masked_select(dets, mask).view(-1, 5)
                        if dets.shape[0] == 0:
                            continue
                        if j:
                            boxes = dets[:, 1:]
                            boxes[:, 0] *= h
                            boxes[:, 1] *= w
                            boxes[:, 2] *= h
                            boxes[:, 3] *= w
                            scores = dets[:, 0].cpu().numpy()
                            for box, score in zip(boxes, scores):
                                y1, x1, y2, x2 = box
                                y1 = int(y1)
                                x1 = int(x1)
                                y2 = int(y2)
                                x2 = int(x2)

                                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                                cv2.putText(ori_img, cfg.VOC_CLASSES[int(j)] + "%.2f" % score, (x1, y1 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 0, 255))
                    cv2.namedWindow('img')
                    cv2.imshow('img', ori_img)
                    k = cv2.waitKey(10000)
                    if k & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        exit()
                cv2.destroyAllWindows()
                exit()

    def eval(self):
                print('evaluation mode...')
                self.SSD.eval()
                print('loading data...')
                dsets = PASCALVOC(img_size=cfg.img_size,
                                  root=cfg.root,
                                  image_sets=cfg.test_sets,
                                  phase='test', mean=cfg.means, std=cfg.std)
                output_dir = cfg.output_dir
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                else:
                    shutil.rmtree(output_dir)
                    os.mkdir(output_dir)

                num_imgs = len(dsets)
                total_time = 0

                det_file = os.path.join(output_dir, 'detections.pkl')
                print('Detecting bounding boxes...')
                all_boxes = [[[] for _ in range(num_imgs)] for _ in range(cfg.num_classes)]

                _t = {'im_detect': Timer(), 'misc': Timer()}

                for i in tqdm(range(num_imgs)):
                    img, target = dsets.__getitem__(i)
                    ori_img = cv2.imread(dsets._imgpath % dsets.img_ids[i])
                    h, w, c = ori_img.shape

                    x = img.unsqueeze(0)
                    x = x.to(self.device)

                    _t['im_detect'].tic()
                    detections = self.SSD(x, 'test')
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

                    print('img-detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_imgs, detect_time))
                    total_time += detect_time

                with open(det_file, 'wb') as f:
                    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
                    f.close()

                print('Saving the results...')
                for cls_ind, cls in enumerate(cfg.labelmap):
                    print('Writing {:s} VOC results file'.format(cls))
                    filename = get_voc_results_file_template('test', cls)
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
                print('average time is {}'.format(float(total_time) / num_imgs))
                do_python_eval(output_dir=output_dir, use_07=True)

    def eval_single(self):
                print('evaluation mode...')
                self.model.eval()

                print('loading data...')
                dsets = PASCALVOC(img_size=cfg.img_size,
                                  root=cfg.root,
                                  image_sets=cfg.test_sets,
                                  phase='val',mean=cfg.means,std=cfg.std)

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
                        boxes[:, 0] = np.maximum(0., boxes[:, 0])
                        boxes[:, 1] = np.maximum(0., boxes[:, 1])
                        boxes[:, 2] = np.minimum(h, boxes[:, 2])
                        boxes[:, 3] = np.minimum(w, boxes[:, 3])
                        scores = dets[:, 0].cpu().numpy()
                        cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                            .astype(np.float32, copy=False)
                        all_boxes[j][i] = cls_dets

                    print('img-detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_imgs, detect_time))

                with open(det_file, 'wb') as f:
                    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                print('average time is {}'.format(float(total_time) / num_imgs))
                print('Saving the results...')
                for cls_ind, cls in enumerate(cfg.labelmap):
                    print('Writing {:s} VOC results file'.format(cls))
                    filename = get_voc_results_file_template('test', cls)
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
                do_python_eval(output_dir=output_dir, use_07=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=None,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--vis', type=bool, default=False, help='view data set')
    parser.add_argument('--mode', type=str, default='', help='eval eval_single or test')
    opt = parser.parse_args()
    if opt.mode == 'eval':
        Trainer(weight_path=opt.weight_path,
                resume=opt.resume,
                gpu_id=opt.gpu_id,
                vis=opt.vis,
                mode=opt.mode).eval()
    elif opt.mode == 'eval_single':
        Trainer(weight_path=opt.weight_path,
                resume=opt.resume,
                gpu_id=opt.gpu_id,
                vis=opt.vis,
                mode=opt.mode).eval_single()
    elif opt.mode == 'test':
        Trainer(weight_path=opt.weight_path,
                resume=opt.resume,
                gpu_id=opt.gpu_id,
                vis=opt.vis,
                mode=opt.mode).test()
    else:
        Trainer(weight_path=opt.weight_path,
                resume=opt.resume,
                gpu_id=opt.gpu_id,
                vis=opt.vis).train()