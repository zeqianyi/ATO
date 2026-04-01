import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import math

from ._base import Distiller


def kd_loss(logits_student_in, logits_teacher_in, temperature, temperature_weight):
    logits_student = logits_student_in
    logits_teacher = logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature_weight**2
    return loss_kd


def kl_div_compute(logits_student_in, logits_teacher_in, temperature):
    logits_student = logits_student_in
    logits_teacher = logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    kl_div = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    return kl_div

def l1_norm(output_teacher_p, output_student_p):
    l1_norm = torch.mean(torch.abs(output_teacher_p - output_student_p))
    return l1_norm


def threshold_low_compute(logits_student_in, logits_teacher_in,T):
    # compute L1 norm
    logits_student = logits_student_in
    logits_teacher = logits_teacher_in
    pred_teacher = F.softmax(logits_teacher / T, dim=1)
    pred_student = F.softmax(logits_student / T, dim=1)
    batch_size = logits_teacher.size(0)

    l1_norm = torch.norm(pred_teacher - pred_student, p=1) / batch_size
    l1_norm_squared = (l1_norm ** 2)
    threshold_low = 0.5 * l1_norm_squared  

    return threshold_low


def threshold_up_compute(logits_student_in, logits_teacher_in,T):
    threshold_up = kl_div_compute(logits_student_in, logits_teacher_in, T)

    return threshold_up


def threshold_compute(logits_student_in, logits_teacher_in, T, lambda_threshold, epoch, idx):
    threshold_up = threshold_up_compute(logits_student_in, logits_teacher_in,T)
    threshold_low = threshold_low_compute(logits_student_in, logits_teacher_in,T)

    threshold = threshold_low + lambda_threshold * (threshold_up - threshold_low)


    return threshold


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.temperature_optimize = self.temperature
        self.temperature_weight = cfg.KD.TEMPERATURE_WEIGHT
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.adaptive_temperature = cfg.EXPERIMENT.ADAPTIVE_TEMPERATURE
        self.temperature_lr = cfg.EXPERIMENT.TEMPERATURE_LR
        self.lambda_threshold = cfg.EXPERIMENT.LAMBDA_THRESHOLD
        self.cfg = cfg

    def forward_train(self, image, target, epoch, idx, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        if self.adaptive_temperature:
            # stage1: fix student, optimize τ
            for param in self.student.parameters():
                param.requires_grad = False
            tau = nn.Parameter(torch.tensor(self.temperature, dtype=torch.float32).cuda())
            optimizer_tau = torch.optim.SGD(
                params=[tau],
                lr=self.temperature_lr,
                momentum=0,
                weight_decay=0,
                nesterov=False,
            )
            threshold = threshold_compute(logits_student.detach(), logits_teacher.detach(), self.temperature, self.lambda_threshold, epoch, idx)
            lr_start = self.temperature_lr



            best_tau = 0
            max_steps = 100  #
            previous_tau = tau.item()
            previous_kl_loss = 0
            for step in range(max_steps):
                current_lr = (1.5 ** step) * lr_start
                for param_group in optimizer_tau.param_groups:
                    param_group['lr'] = current_lr
                kl_loss_tau = kl_div_compute(logits_student.detach(), logits_teacher.detach(), tau)
                best_tau = previous_tau

                # If kl_loss_tau is less than threshold, revert to the previous tau and break
                if kl_loss_tau.item() < threshold.item():
                    best_tau = previous_tau
                    break

                if previous_kl_loss != 0 and kl_loss_tau >= previous_kl_loss - 1e-5:
                    best_tau = previous_tau
                    break

                # Save the current tau and kl_loss for the next iteration
                previous_tau = tau.item()
                previous_kl_loss = kl_loss_tau.item()

                optimizer_tau.zero_grad()
                kl_loss_tau.backward()
                optimizer_tau.step()

            # stage2：fix τ, distill student
            for param in self.student.parameters():
                param.requires_grad = True

            self.temperature_optimize = best_tau
            if idx == 781:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_msg_tau = '[{0}] Epoch: [{1}/240] tau_new: {tau_new:.3f}'.format(
                    current_time, epoch, tau_new=self.temperature_optimize)
                with open(self.cfg.EXPERIMENT.LOG_FILE_PATH, "a") as log_file:
                    log_file.write(log_msg_tau + "\n")

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature_optimize, self.temperature_weight
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
