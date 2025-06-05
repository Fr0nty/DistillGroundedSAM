from quick_eval import compare_models

# Compare different training runs
models = [
    "enhanced_dinov2_student_cifar10.pth",
    "enhanced_dinov2_student_food101.pth"
]

results = compare_models(models, dataset_choice="food101")