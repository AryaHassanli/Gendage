import torchvision.transforms as transforms

options = {
    "datasetsDir":  "D:/MsThesis/datasets",
    "outputDir":    "output",

    "nets":         ["simClass", "simClass"],
    "datasets":     ["AgeDB", "UTKFace"],
    "features":     ["gender", "age"],
    "numOfClasses": [2, 120],

    "preload":      False,

    "splitSize":    [0.7, 0.2, 0.1],
    "batchSize":    32,
    "epochs":       20,
    "lr":           0.005,

    "preTransforms": transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Pad(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
