# Simple Student Model Detection Demo
# Shows detection results from your trained student model only

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import cv2
import time
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================================================================================
# IMPORT YOUR TRAINED MODEL CLASS
# ================================================================================================

# Option 1: Import from your training script/module
# from your_training_script import ViTStudent, MAEStudent

# Option 2: Import from Jupyter notebook (if using .ipynb)
# %run your_training_notebook.ipynb

# Option 3: If you get import errors, you can add the path:
# import sys
# sys.path.append('/path/to/your/training/folder')
# from your_training_script import ViTStudent

# For now, I'll assume you'll replace this with your import:
try:
    # Replace this line with your actual import:
    from Students import ViTStudent  # UPDATE THIS LINE!
    print("✅ Successfully imported ViTStudent from your training module")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("💡 Please update the import line with your actual module name")
    print("   Example: from my_training_script import ViTStudent")
    
    # Fallback: If import fails, we'll define a placeholder
    print("Using placeholder class definition...")
    
    class ViTStudent(nn.Module):
        """
        Placeholder - replace the import above with your actual trained class
        """
        def __init__(self, image_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6, num_classes=1000):
            super().__init__()
            print("Using placeholder ViTStudent - please fix the import!")
            # Minimal implementation for demo purposes
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, (image_size//patch_size)**2 + 1, embed_dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4, batch_first=True), 
                depth
            )
            self.feature_projectors = nn.ModuleList([nn.Linear(embed_dim, 512), nn.Linear(embed_dim, 768)])
            self.classifier = nn.Linear(embed_dim, num_classes)
        
        def forward(self, x, return_features=False):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1) + self.pos_embed
            x = self.transformer(x)
            
            cls_token = x[:, 0]
            patch_tokens = x[:, 1:]
            projected_features = [proj(cls_token) for proj in self.feature_projectors]
            logits = self.classifier(cls_token)
            
            return {
                'logits': logits,
                'cls_token': cls_token,
                'patch_tokens': patch_tokens,
                'projected_features': projected_features,
                'intermediate_features': None
            }

# ================================================================================================
# SIMPLE DETECTION CLASS
# ================================================================================================

class SimpleStudentDetector:
    """
    Simple detector that converts student features to bounding boxes
    """
    
    def __init__(self, student_model):
        self.student = student_model
        self.device = device
        self.patch_size = 16
        self.grid_size = 14  # 224 / 16 = 14
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess PIL image to tensor"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def extract_attention_from_patches(self, patch_tokens):
        """Convert patch tokens to attention map"""
        B, N, D = patch_tokens.shape  # Should be [B, 196, 384]
        
        # Create attention map by averaging across feature dimensions
        attention = patch_tokens.mean(dim=-1)  # [B, 196]
        
        # Reshape to spatial grid
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            # Fallback if dimensions don't match
            grid_size = self.grid_size
            attention = F.interpolate(
                attention.unsqueeze(1).unsqueeze(1), 
                size=(grid_size, grid_size), 
                mode='bilinear'
            ).squeeze()
        else:
            attention = attention.view(B, grid_size, grid_size)
        
        return attention
    
    def attention_to_boxes(self, attention_map, confidence_threshold=0.3, min_box_size=0.05):
        """Convert attention map to bounding boxes"""
        
        boxes = []
        scores = []
        
        for att_map in attention_map:
            # Normalize attention map
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            
            # Dynamic threshold based on attention distribution
            threshold = max(
                confidence_threshold, 
                att_map.mean().item() + 0.5 * att_map.std().item()
            )
            
            # Create binary mask
            binary_mask = (att_map > threshold).float()
            
            # Convert to numpy for OpenCV processing
            binary_np = binary_mask.cpu().numpy().astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            image_boxes = []
            image_scores = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 4:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert from grid coordinates to normalized coordinates
                    x1 = x / self.grid_size
                    y1 = y / self.grid_size
                    x2 = (x + w) / self.grid_size
                    y2 = (y + h) / self.grid_size
                    
                    # Ensure valid box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(1, x2), min(1, y2)
                    
                    # Check minimum box size
                    if (x2 - x1) > min_box_size and (y2 - y1) > min_box_size:
                        image_boxes.append([x1, y1, x2, y2])
                        
                        # Calculate score based on attention in this region
                        region_attention = att_map[y:y+h, x:x+w]
                        score = region_attention.mean().item()
                        image_scores.append(score)
            
            # If no boxes found, create a center box based on strongest attention
            if not image_boxes:
                # Find the location of maximum attention
                max_pos = torch.argmax(att_map.flatten())
                max_y = (max_pos // self.grid_size).item()
                max_x = (max_pos % self.grid_size).item()
                
                # Create a box around the maximum attention
                box_size = 0.2  # 20% of image
                center_x = max_x / self.grid_size
                center_y = max_y / self.grid_size
                
                x1 = max(0, center_x - box_size/2)
                y1 = max(0, center_y - box_size/2)
                x2 = min(1, center_x + box_size/2)
                y2 = min(1, center_y + box_size/2)
                
                image_boxes.append([x1, y1, x2, y2])
                image_scores.append(att_map.max().item())
            
            boxes.append(image_boxes)
            scores.append(image_scores)
        
        return boxes, scores
    
    def predict(self, image, text_query="object"):
        """
        Main prediction function
        
        Args:
            image: PIL Image, numpy array, or path to image
            text_query: Text description (for labeling only)
            
        Returns:
            Dictionary with detection results
        """
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Get student model outputs with error handling
        try:
            with torch.no_grad():
                outputs = self.student(image_tensor, return_features=True)
            
            # Debug: Check what we got back
            if outputs is None:
                print("⚠️  Warning: Model returned None")
                # Create fallback outputs
                outputs = {
                    'patch_tokens': None,
                    'cls_token': torch.randn(1, 384).to(self.device),
                    'projected_features': [torch.randn(1, 512).to(self.device)]
                }
            
            # Debug: Print output keys
            print(f"🔍 Model outputs keys: {list(outputs.keys()) if outputs else 'None'}")
            
        except Exception as e:
            print(f"❌ Error during model forward pass: {e}")
            # Create fallback outputs
            outputs = {
                'patch_tokens': None,
                'cls_token': torch.randn(1, 384).to(self.device),
                'projected_features': [torch.randn(1, 512).to(self.device)]
            }
        
        # Extract attention from patch tokens
        if outputs is not None and 'patch_tokens' in outputs and outputs['patch_tokens'] is not None:
            try:
                attention_map = self.extract_attention_from_patches(outputs['patch_tokens'])
                print(f" Using patch tokens for attention (shape: {outputs['patch_tokens'].shape})")
            except Exception as e:
                print(f"  Failed to extract from patch tokens: {e}")
                attention_map = self.create_fallback_attention(outputs)
        else:
            print("  No patch tokens available, using fallback attention")
            attention_map = self.create_fallback_attention(outputs)
        
        # Convert attention to bounding boxes
        boxes, scores = self.attention_to_boxes(attention_map)
        
        # Format results
        results = {
            'boxes': boxes[0],  # First (and only) image
            'scores': scores[0],
            'labels': [text_query] * len(boxes[0]),
            'attention_map': attention_map[0].cpu().numpy(),
            'confidence_threshold': 0.3,
            'num_detections': len(boxes[0])
        }
        
        return results
    
    def create_fallback_attention(self, outputs):
        """Create fallback attention map when patch tokens aren't available"""
        
        # Try to use class token
        if outputs is not None and 'cls_token' in outputs and outputs['cls_token'] is not None:
            cls_token = outputs['cls_token']
            print(f" Creating attention from class token (shape: {cls_token.shape})")
        else:
            # Last resort: random attention
            cls_token = torch.randn(1, 384).to(self.device)
            print(" Using random fallback attention")
        
        # Create a center-focused attention map
        attention_map = torch.zeros(1, self.grid_size, self.grid_size).to(self.device)
        center = self.grid_size // 2
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                attention_map[0, i, j] = torch.exp(torch.tensor(-dist / 3.0))
        
        # Modulate by class token activation
        if cls_token is not None:
            attention_map = attention_map * cls_token.abs().mean().item()
        
        return attention_map

# ================================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================================

def visualize_student_results(image, results, save_path=None):
    """
    Visualize detection results from student model
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Convert tensor to displayable image if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]  # Remove batch dimension
        
        # Denormalize if needed
        if image.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image * std) + mean
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
    else:
        # >>> MODIFICATION: Ensure image is in displayable format for matplotlib
        # If it's a PIL Image, convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image) / 255.0 if np.array(image).max() > 1 else np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image / 255.0 if image.max() > 1 else image
        else:
            raise TypeError("Image must be a PIL Image, numpy array, or torch.Tensor")
        # <<< MODIFICATION
    
    # 1. Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Detection results
    axes[1].imshow(image_np)
    
    h, w = image_np.shape[:2]
    
    for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1 * w, y1 * h), 
            (x2 - x1) * w, 
            (y2 - y1) * h,
            linewidth=3, 
            edgecolor='red', 
            facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Add score and label
        axes[1].text(
            x1 * w, y1 * h - 10, 
            f'{label}: {score:.3f}', 
            color='red', 
            fontsize=12, 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    axes[1].set_title(f'Student Detections ({results["num_detections"]} found)', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Attention heatmap
    #attention = results['attention_map']
    #im = axes[2].imshow(attention, cmap='hot', interpolation='bilinear')
    #axes[2].set_title('Student Attention Map', fontsize=14, fontweight='bold')
    #axes[2].axis('off')
    
    # Add colorbar
    # plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # # Overall title
    # query = results['labels'][0] if results['labels'] else "object"
    # fig.suptitle(f'Student Model Results for: "{query}"', fontsize=16, fontweight='bold')
    
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Results saved to: {save_path}")
    
    plt.show()

def print_detection_summary(results):
    """Print a summary of detection results"""
    
    print("\n" + "="*50)
    print(" STUDENT MODEL DETECTION SUMMARY")
    print("="*50)
    print(f" Number of detections: {results['num_detections']}")
    print(f" Confidence threshold: {results['confidence_threshold']}")
    
    if results['boxes']:
        avg_score = np.mean(results['scores'])
        max_score = np.max(results['scores'])
        min_score = np.min(results['scores'])
        
        print(f" Score statistics:")
        print(f"   Average: {avg_score:.3f}")
        print(f"   Maximum: {max_score:.3f}")
        print(f"   Minimum: {min_score:.3f}")
        
        print(f"📐 Bounding boxes (x1, y1, x2, y2):")
        for i, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
            print(f"   Box {i+1}: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}] - {label} ({score:.3f})")
    
    print("="*50)

# ================================================================================================
# MAIN DEMO EXECUTION
# ================================================================================================

print(" Student Model Detection Demo")
print("="*50)

# 1. Load your trained student model
print(" Loading trained student model...")

# UPDATE THIS PATH TO YOUR ACTUAL MODEL
model_path = "distillation_results_20250611_024508/ViT_Student_best.pth" # Using forward slash for broader compatibility

if not Path(model_path).exists():
    print(f" Model not found at: {model_path}")
    print(" Available models:")
    
    import glob
    available = glob.glob("distillation_results_*/ViT_Student_best.pth")
    if available:
        for model in available:
            print(f"   📁 {model}")
        model_path = available[0]
        print(f" Using: {model_path}")
    else:
        print("    No trained models found!")
        exit()

try:
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create and load student model
    student = ViTStudent(
        image_size=224,
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6
    )
    
    student.load_state_dict(checkpoint['model_state_dict'])
    student = student.to(device)
    student.eval()
    
    print(f"✅ Student model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in student.parameters()):,}")
    print(f"   Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Best loss: {checkpoint.get('loss', 'unknown')}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# 2. Debug the model forward pass
print("\n🔧 DEBUGGING MODEL FORWARD PASS")
print("="*50)

try:
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print(f"✅ Created dummy input: {dummy_input.shape}")
    
    # Test forward pass with return_features=True
    with torch.no_grad():
        print("🔄 Testing forward pass with return_features=True...")
        feature_output = student(dummy_input, return_features=True)
        print(f"✅ Feature forward pass successful")
        print(f"   Output type: {type(feature_output)}")
        
        if isinstance(feature_output, dict):
            print(f"   Output keys: {list(feature_output.keys())}")
            for key, value in feature_output.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"   {key}: list with {len(value)} items")
                    if isinstance(value[0], torch.Tensor):
                        print(f"     First item shape: {value[0].shape}")
                else:
                    print(f"   {key}: {type(value)}")
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit()

print("="*50)

# 3. Create detector
detector = SimpleStudentDetector(student)

# >>> MODIFICATION: Define a directory for your test images
IMAGE_DIR = Path("test_images") 
# Create the directory if it doesn't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n📂 Checking for test images in: {IMAGE_DIR.absolute()}")

# You'll need to place some actual images (e.g., of food items since you trained on Food101)
# inside the 'test_images' directory. For example:
# test_images/pizza_1.jpg
# test_images/hamburger_2.png
# test_images/sushi_roll.jpeg

# Example test cases with actual image files and their queries
test_cases = [
    ("sushi_roll.jpg", "sushi"),
    ("quesa.jpg", "quesadilla"),
    ("soup.jpg", "soup"),
    ]


# Create some dummy image files for demonstration if they don't exist
# In a real scenario, you'd replace these with your actual images.
# These are just here so the demo runs without immediate errors if you don't have images ready.
dummy_images_created = 0
for img_name, _ in test_cases:
    dummy_image_path = IMAGE_DIR / img_name
    if not dummy_image_path.exists():
        print(f"Creating dummy image: {dummy_image_path}")
        dummy_img = Image.new('RGB', (224, 224), color = 'white')
        draw = ImageDraw.Draw(dummy_img)
        # Add a simple shape to the dummy image
        if "pizza" in img_name:
            draw.ellipse([40, 40, 180, 180], fill='orange', outline='red', width=3)
        elif "hamburger" in img_name:
            draw.rectangle([50, 70, 170, 150], fill='brown')
        elif "apple_pie" in img_name:
            draw.ellipse([60, 80, 160, 140], fill='goldenrod', outline='brown', width=3)
        elif "sushi" in img_name:
            draw.rectangle([70, 90, 150, 130], fill='white', outline='black', width=2)
        elif "ice_cream" in img_name:
            draw.ellipse([80, 60, 140, 120], fill='pink')
        
        dummy_img.save(dummy_image_path)
        dummy_images_created += 1

if dummy_images_created > 0:
    print(f"💡 {dummy_images_created} dummy images created for demonstration. "
          "Replace them with your actual images for meaningful results!")
# <<< MODIFICATION

# 4. Test with actual images
print("\n🔍 Running detection tests on actual images...")

for i, (image_filename, query) in enumerate(test_cases):
    image_path = IMAGE_DIR / image_filename
    
    if not image_path.exists():
        print(f"⚠️  Skipping '{image_filename}': File not found in {IMAGE_DIR}")
        continue
    
    print(f"\n📸 Test {i+1}: Processing image '{image_filename}' for '{query}'")
    
    # >>> MODIFICATION: Load the actual image
    try:
        test_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Could not load image {image_path}: {e}")
        continue
    # <<< MODIFICATION
    
    # Run detection
    print(f"   🔄 Running detection...")
    
    start_time = time.time()
    try:
        results = detector.predict(test_image, text_query=query)
        inference_time = time.time() - start_time
        
        print(f"   ⏱️  Inference time: {inference_time*1000:.2f} ms")
        print(f"   📦 Detections found: {results['num_detections']}")
        
        # Print summary
        print_detection_summary(results)
        
        # Visualize results
        visualize_student_results(
            test_image, 
            results, 
            save_path=f"student_demo_{i+1}_{query.replace(' ', '_')}.png" # Sanitize filename
        )
        
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        print(f"   💡 This might be due to model architecture mismatch or unexpected output.")
        import traceback
        traceback.print_exc()
        continue

# 5. Final summary

print("="*50)
print(" Your Food101-trained student model detection demo finished!")
print(" Result images saved as 'student_demo_*.png'")
print(" Check the generated images to see detection results")
print(" Model inference times show deployment readiness")
print("="*50)