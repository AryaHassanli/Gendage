import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN

from config import config
from helpers import eval
from helpers import parseArguments
from helpers import demo

args = parseArguments.parse('run')
if args.gradient:
    config.set(datasetDir='/storage/datasets',
               outputDir='/artifacts')
else:
    config.set(datasetDir="D:\\MsThesis\\datasets",
               outputDir='output')

gendage = eval.genderAndAge
newAge = eval.newAge


def main():
    """
    demo.process(online=1,
         labelGenerators=[newAge],
         inputFile='D:/MsThesis/inputSamples/Recording.mp4',
         outputFile='output/result.avi',
         detectionFPS=5,
         device=config.device)
    """

    image = Image.open('inputSamples/105_PaulAnka_18_m.jpg')
    image = np.array(image)
    face = image
    mtcnn = MTCNN(keep_all=True, device=config.device)
    boxes, _ = mtcnn.detect(image)
    faces = []
    if boxes is None:
        return faces
    for box in boxes:
        box = box.astype(int)
        x1, y1, x2, y2 = box
        x1 = min(image.shape[1], max(0, x1))
        x2 = min(image.shape[1], max(0, x2))
        y1 = min(image.shape[0], max(0, y1))
        y2 = min(image.shape[0], max(0, y2))
        faces.append({
            'box': (x1, y1, x2, y2),
            'labels': [],
            'image': image[y1:y2, x1:x2]
        })
    face = faces[0]['image']
    age = eval.newAge(face)
    print(age)
    """
    """

if __name__ == '__main__':
    main()
