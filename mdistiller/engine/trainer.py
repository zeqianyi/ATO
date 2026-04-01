import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from datetime import datetime
from mdistiller.engine.cfg import show_cfg
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
)


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        #log_name
        if cfg.DISTILLER.TYPE == 'KD':
            self.log_name = 'S:{}_T:{}_{}_{}_temp:{}_temp_weight:{}_lambda:{}_lr:{}_ce:{}_kl:{}_seed:{}'.format(
                cfg.DISTILLER.TEACHER, cfg.DISTILLER.STUDENT, cfg.DATASET.TYPE, cfg.DISTILLER.TYPE, cfg.KD.TEMPERATURE,
                cfg.KD.TEMPERATURE_WEIGHT,cfg.EXPERIMENT.LAMBDA_THRESHOLD, cfg.EXPERIMENT.TEMPERATURE_LR, cfg.KD.LOSS.CE_WEIGHT, cfg.KD.LOSS.KD_WEIGHT, cfg.EXPERIMENT.SEED)
        elif cfg.DISTILLER.TYPE == 'DKD':
            self.log_name = 'S:{}_T:{}_{}_{}_temp:{}_temp_weight:{}_lambda:{}_lr:{}_ce:{}_alpha:{}_beta:{}_seed:{}'.format(
                cfg.DISTILLER.TEACHER, cfg.DISTILLER.STUDENT, cfg.DATASET.TYPE, cfg.DISTILLER.TYPE, cfg.DKD.T,
                cfg.DKD.T_WEIGHT, cfg.EXPERIMENT.LAMBDA_THRESHOLD, cfg.EXPERIMENT.TEMPERATURE_LR, cfg.DKD.CE_WEIGHT, cfg.DKD.ALPHA, cfg.DKD.BETA, cfg.EXPERIMENT.SEED)
        elif cfg.DISTILLER.TYPE == 'ABKD':
            self.log_name = 'S:{}_T:{}_{}_{}_temp:{}_lambda:{}_lr:{}_ce:{}_kl:{}_alpha:{}_beta:{}_seed:{}'.format(
                cfg.DISTILLER.TEACHER, cfg.DISTILLER.STUDENT, cfg.DATASET.TYPE, cfg.DISTILLER.TYPE, cfg.ABKD.TEMPERATURE,
                cfg.EXPERIMENT.LAMBDA_THRESHOLD, cfg.EXPERIMENT.TEMPERATURE_LR, cfg.ABKD.LOSS.CE_WEIGHT,cfg.ABKD.LOSS.KD_WEIGHT, cfg.ABKD.ALPHA, cfg.ABKD.BETA, cfg.EXPERIMENT.SEED)
        elif cfg.DISTILLER.TYPE == 'LSKD':
            self.log_name = 'S:{}_T:{}_{}_{}_temp:{}_temp_weight:{}_lambda:{}_lr:{}_ce:{}_kl:{}_seed:{}_logit_stand'.format(
                cfg.DISTILLER.TEACHER, cfg.DISTILLER.STUDENT, cfg.DATASET.TYPE, cfg.DISTILLER.TYPE, cfg.LSKD.TEMPERATURE,
                cfg.LSKD.TEMPERATURE_WEIGHT,cfg.EXPERIMENT.LAMBDA_THRESHOLD, cfg.EXPERIMENT.TEMPERATURE_LR, cfg.LSKD.LOSS.CE_WEIGHT, cfg.LSKD.LOSS.KD_WEIGHT, cfg.EXPERIMENT.SEED)

        self.log_file_path = os.path.join(self.log_path, self.log_name + ".txt")
        self.log_save_ckpt_path = os.path.join(self.log_path, self.log_name + "_student_best.pth")
        cfg.EXPERIMENT.LOG_FILE_PATH = self.log_file_path

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        cfg.freeze()
        cfg_str = show_cfg(cfg)

        with open(self.cfg.EXPERIMENT.LOG_FILE_PATH, "a") as log_file:
            log_file.write(cfg_str + "\n")



    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        log_msg_save = 'Best Acc@1 {top1:.3f}'.format(top1=self.best_acc)
        print(log_msg_save)
        with open(self.log_file_path, "a") as log_file:
            log_file.write(log_msg_save + "\n")

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            train_acc, train_acc_top5, train_loss = self.train_iter(data, epoch, idx, train_meters)
        current_time_train = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg_train = '[{0}] Epoch: [{1}/240] Train Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(current_time_train, epoch, top1=train_acc, top5=train_acc_top5)
        print(log_msg_train)
        with open(self.log_file_path, "a") as log_file:
            log_file.write(log_msg_train + "\n")

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller,epoch)

        current_time_test = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg_test = '[{0}] Epoch: [{1}/240] Test Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(current_time_test, epoch, top1=test_acc, top5=test_acc_top5)
        print(log_msg_test )

        with open(self.log_file_path, "a") as log_file:
            log_file.write(log_msg_test + "\n")

        # saving checkpoint
        student_state = {"model": self.distiller.module.student.state_dict()}
        # update the best
        if test_acc >= self.best_acc:
            self.best_acc = test_acc
            save_checkpoint(student_state, self.log_save_ckpt_path)

            log_msg_save = '[{0}] Epoch: [{1}/240] saving the best model! Best Acc@1 {top1:.3f}'.format(current_time_test,epoch, top1=self.best_acc)
            print(log_msg_save)
            with open(self.log_file_path, "a") as log_file:
                log_file.write(log_msg_save + "\n")

    def train_iter(self, data, epoch, idx, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch, idx=idx)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        return train_meters["top1"].avg, train_meters["top5"].avg, train_meters["losses"].avg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class AugTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image_weak, image_strong = image
        image_weak, image_strong = image_weak.float(), image_strong.float()
        image_weak, image_strong = image_weak.cuda(non_blocking=True), image_strong.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image_weak=image_weak, image_strong=image_strong, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image_weak.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
