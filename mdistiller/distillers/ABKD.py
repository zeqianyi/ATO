import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def abkd_loss(logits_student_in,logits_teacher_in,temperature, alpha, beta):
    s_logit = logits_student_in
    t_logit = logits_teacher_in

    teacher_probs = F.softmax(t_logit / temperature, dim=1)
    student_probs = F.softmax(s_logit / temperature, dim=1)

    # Create inf_mask to handle infinite logits
    inf_mask = torch.isinf(s_logit) | torch.isinf(t_logit)

    # Special case when alpha = 0 and beta = 0
    if alpha == 0 and beta == 0:
        log_diff = torch.log(student_probs) - torch.log(teacher_probs)
        log_diff = torch.masked_fill(log_diff, inf_mask, 0)  # Handle infinities
        divergence = 0.5 * torch.sum(log_diff ** 2, dim=1)  # Use L2 divergence
    elif alpha == 0:
        # Case where alpha = 0
        q_beta = torch.pow(student_probs, beta)
        p_beta = torch.pow(teacher_probs, beta)
        q_beta = torch.masked_fill(q_beta, inf_mask, 0)
        p_beta = torch.masked_fill(p_beta, inf_mask, 0)
        likeli_ratio = q_beta / p_beta
        likeli_ratio = torch.masked_fill(likeli_ratio, torch.isnan(likeli_ratio), 0)
        divergence = (1 / beta) * torch.sum(
            q_beta * torch.log(likeli_ratio) - q_beta + p_beta,
            dim=1,
        )
    elif beta == 0:
        # Case where beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.pow(student_probs, alpha)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        divergence = (1 / alpha) * torch.sum(
            p_alpha * torch.log(p_alpha / q_alpha) - p_alpha + q_alpha,
            dim=1,
        )
    elif alpha + beta == 0:
        # Case where alpha + beta = 0
        p_alpha = torch.pow(teacher_probs, alpha)
        q_alpha = torch.pow(student_probs, alpha)
        p_alpha = torch.masked_fill(p_alpha, inf_mask, 0)
        q_alpha = torch.masked_fill(q_alpha, inf_mask, 0)
        divergence = torch.sum(
            (1 / alpha) * (torch.log(q_alpha / p_alpha) + (q_alpha / p_alpha).reciprocal() - 1),
            dim=1,
        )
    else:
        # General case
        p_alpha = torch.pow(teacher_probs, alpha)
        q_beta = torch.pow(student_probs, beta)
        p_alpha_beta = torch.pow(teacher_probs, alpha + beta)
        q_alpha_beta = torch.pow(student_probs, alpha + beta)

        # First term: - ∑ p_T^α * q_S^β
        first_term = p_alpha * q_beta
        # Second term: α / (α + β) * ∑ p_T^(α + β)
        second_term = (alpha / (alpha + beta)) * p_alpha_beta
        # Third term: β / (α + β) * ∑ q_S^(α + β)
        third_term = (beta / (alpha + beta)) * q_alpha_beta

        # Mask invalid values
        first_term = torch.masked_fill(first_term, inf_mask, 0)
        second_term = torch.masked_fill(second_term, inf_mask, 0)
        third_term = torch.masked_fill(third_term, inf_mask, 0)

        # Compute divergence
        divergence = -torch.sum(first_term - second_term - third_term, dim=1) / (alpha * beta)
    return divergence.mean()

class ABKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(ABKD, self).__init__(student, teacher)
        self.temperature = cfg.ABKD.TEMPERATURE
        self.ce_loss_weight = cfg.ABKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.ABKD.LOSS.KD_WEIGHT
        self.alpha = cfg.ABKD.ALPHA
        self.beta = cfg.ABKD.BETA

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * abkd_loss(
            logits_student, logits_teacher, self.temperature, self.alpha, self.beta
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
