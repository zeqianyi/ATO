import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import random

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


def main(cfg, resume, opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    set_seed(cfg.EXPERIMENT.SEED)
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    #show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == 'MLKD':
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    # cfg.freeze()
    # show_cfg(cfg)
    trainer.train(resume=resume)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--logit-stand", action="store_true")
    parser.add_argument("--adaptive-temperature", action="store_true")
    parser.add_argument("--base-temp", type=float, default=2)
    parser.add_argument("--kd-weight", type=float, default=9)
    parser.add_argument("--lambda_threshold", type=float, default=0.9)
    parser.add_argument("--temperature_lr", type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.8, help='ABKD alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='ABKD beta')
    parser.add_argument('--seed', type=int, default=1, help='seed id')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)


    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.EXPERIMENT.SEED = args.seed
    cfg.EXPERIMENT.LAMBDA_THRESHOLD = args.lambda_threshold
    cfg.EXPERIMENT.TEMPERATURE_LR = args.temperature_lr

    if args.adaptive_temperature and cfg.DISTILLER.TYPE in ['KD','DKD','ABKD','LSKD']:
        cfg.EXPERIMENT.ADAPTIVE_TEMPERATURE = True
        if cfg.DISTILLER.TYPE == 'KD':
            cfg.KD.TEMPERATURE = args.base_temp
        elif cfg.DISTILLER.TYPE == 'DKD':
            cfg.DKD.T = args.base_temp
        elif cfg.DISTILLER.TYPE == 'ABKD':
            cfg.ABKD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.ABKD.TEMPERATURE = args.base_temp
            cfg.ABKD.ALPHA = args.alpha
            cfg.ABKD.BETA = args.beta
        elif cfg.DISTILLER.TYPE == 'LSKD' and args.logit_stand:
            cfg.EXPERIMENT.LOGIT_STAND = True
            cfg.LSKD.LOSS.KD_WEIGHT = args.kd_weight
            cfg.LSKD.TEMPERATURE = args.base_temp

    #cfg.freeze()
    main(cfg, args.resume, args.opts)
