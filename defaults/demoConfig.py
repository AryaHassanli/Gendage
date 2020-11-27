import torchvision.transforms as transforms

options = {
  "datasetsDir":    "datasets",
  "outputDir":      "output",
  "preTransforms": transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Pad(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
}
