"""Conservative, configurable transforms for ordinary smartphone photographs."""
from __future__ import annotations
from torchvision import transforms as T

def build_transforms(image_size: int = 224, training: bool = False, augmentation: dict | None = None):
    """Create stochastic training or deterministic evaluation transforms."""
    augmentation = augmentation or {}
    if not training:
        return T.Compose([T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size), T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    steps = [T.RandomResizedCrop(image_size, scale=tuple(augmentation.get("crop_scale", [.75, 1.])), ratio=(.9, 1.1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(augmentation.get("rotation_degrees", 20)), T.RandomAffine(0, translate=tuple(augmentation.get("translate", [.05, .05])), scale=tuple(augmentation.get("scale", [.95, 1.05]))), T.ColorJitter(brightness=augmentation.get("brightness", .10), contrast=augmentation.get("contrast", .10), saturation=augmentation.get("saturation", .08))]
    if augmentation.get("blur", False): steps.append(T.RandomApply([T.GaussianBlur(3, (.1, 1.))], p=.15))
    steps += [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    if augmentation.get("erasing", False): steps.append(T.RandomErasing(p=.15, scale=(.01, .04)))
    return T.Compose(steps)
