import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from timm import create_model
from vit_student import SmallViT
from loss import cosine_distillation_loss
import torch.nn as nn
import torch.optim as optim

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

# ---------- Datasets ----------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# -------------------------------

# ---------- Teacher ----------
teacher = create_model('vit_base_patch16_224_dino', pretrained=True)
teacher.head = nn.Identity()
teacher.to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
# -----------------------------

# ---------- Student ----------
student = SmallViT(embed_dim=student_dim).to(device)
projection = nn.Linear(student_dim, teacher_dim).to(device) if project_to_teacher_dim else nn.Identity().to(device)
optimizer = optim.Adam(list(student.parameters()) + list(projection.parameters()), lr=learning_rate)
# ------------------------------

# ---------- Training Loop ----------
for epoch in range(num_epochs):
    student.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)

        with torch.no_grad():
            teacher_out = teacher(images)

        student_out = student(images)
        student_out = projection(student_out)

        loss = cosine_distillation_loss(student_out, teacher_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
# ----------------------------------
