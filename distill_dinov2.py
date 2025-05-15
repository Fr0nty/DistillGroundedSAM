import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from vit_student import SmallViT
from loss import cosine_distillation_loss
import torch.nn as nn
import torch.optim as optim
from data.hf_cifar import get_cifar_dataloader

def get_actual_output_dim(teacher, img_size, device):
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        actual_output = teacher(dummy_input)
        return actual_output.shape[-1]

def main():
    # ---------- Config ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    student_dim = 192
    teacher_dim = 768
    project_to_teacher_dim = True
    # ----------------------------

    # ---------- Transforms ----------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    # -------------------------------

    # ---------- Datasets ----------
    train_loader = get_cifar_dataloader(split="train[:10%]", batch_size=batch_size)
    val_loader = get_cifar_dataloader(split="test[:10%]", batch_size=batch_size, shuffle=False)
    # -------------------------------

    # ---------- Teacher Model ----------
    teacher = create_model('vit_base_patch16_224_dino', pretrained=True)
    teacher.head = nn.Identity()
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    actual_output_dim = get_actual_output_dim(teacher, img_size, device)
    # -----------------------------------

    # ---------- Student Model ----------
    student = SmallViT(embed_dim=student_dim, actual_output_dim=actual_output_dim).to(device)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
        dummy_output = student(dummy_input)
        print("Shape of student output:", dummy_output.shape)
    projection = nn.Linear(actual_output_dim, teacher_dim) if project_to_teacher_dim else nn.Identity().to(device)
    optimizer = optim.Adam(list(student.parameters()) + list(projection.parameters()), lr=learning_rate)
    # -----------------------------------

    # ---------- Training Loop ----------
    for epoch in range(num_epochs):
        student.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            # Forward through teacher (no grad)
            with torch.no_grad():
                teacher_out = teacher(images)  # shape: [B, D_teacher]

            # Forward through student
            student_out = student(images)     # expected shape: [B, D_student]
            student_out = student_out.mean(dim=1)  # global average pooling over sequence (N) → [B, D_student]
            student_out = projection(student_out)  # shape: [B, D_teacher]

            # Compute loss
            loss = cosine_distillation_loss(student_out, teacher_out)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # -----------------------------------

if __name__ == '__main__':
    main()
