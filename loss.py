import torch.nn.functional as F

def cosine_distillation_loss(student_embed, teacher_embed):
    student_embed = F.normalize(student_embed, dim=-1)
    teacher_embed = F.normalize(teacher_embed, dim=-1)
    return 1 - (student_embed * teacher_embed).sum(dim=-1).mean()
