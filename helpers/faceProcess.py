from facenet_pytorch import MTCNN
from PIL import Image
from config import config


def detect(frame):
    mtcnn = MTCNN(keep_all=True, device=config.device)
    boxes, _ = mtcnn.detect(frame)
    faces = []
    if boxes is None:
        return faces
    for box in boxes:
        box = box.astype(int)
        x1, y1, x2, y2 = box
        x1 = min(frame.shape[1], max(0, x1))
        x2 = min(frame.shape[1], max(0, x2))
        y1 = min(frame.shape[0], max(0, y1))
        y2 = min(frame.shape[0], max(0, y2))
        faces.append({
            'box': (x1, y1, x2, y2),
            'labels': [],
            'image': Image.fromarray(frame[y1:y2, x1:x2])
        })
    return faces


def align(faceImage):
    x, y = faceImage.size
    size = max(x, y)
    faceSq = Image.new('RGB', (size, size), (0, 0, 0))
    faceSq.paste(faceImage, (int((size - x) / 2), int((size - y) / 2)))

    return faceSq
