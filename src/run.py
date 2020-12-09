import os

import cv2
import numpy
from PIL import Image

from src.helpers import eval, faceProcess, getClassifier, getEncoder
from src.helpers.config import config


def main():
    path = config.encoder_pretrain
    encoder = getEncoder.get(config.encoder, pretrained=path).to(config.device)
    encoder.eval()

    classifiers = []
    for i, feature in enumerate(config.features):
        classifiers.append(getClassifier.get(config.classifiers[i], pretrained=config.classifier_pretrain[i],
                                             inputSize=512, numOfClasses=config.num_classes[i]).to(config.device))

    frame = Image.open(config.input_file)
    frame = numpy.array(frame)
    faces = faceProcess.detect(frame)
    if faces:
        for i, face in enumerate(faces):
            faces[i]['image'] = faceProcess.align(face['image'])
        faces = [face for face in faces if face['image'] is not None]

        for face in faces:
            face['labels'] += label_generator(face['image'], encoder, config.features, classifiers)

        for face in faces:
            put_labels(frame, face)
    frame = Image.fromarray(frame)

    frame.save(os.path.join(config.abs_output_dir, "labeled_" + os.path.basename(config.input_file)))


def label_generator(face, encoder, features, classifiers):
    out = eval.encoder_multitask(face, features, encoder, classifiers)
    output_list = []
    for feature in features:
        if feature == 'age':
            output_list.append(str(round(
                out[feature],
                2)))
            pass
        elif feature == 'gender':
            output_list.append('male' if out['gender'] < 0.5 else 'female')
            pass
        else:
            output_list.append(str(round(
                out[feature],
                2)))
    return output_list


def put_labels(frame, face):
    # TODO: Add padding
    x1, y1, x2, y2 = face['box']
    labels = face['labels']
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.putText(frame, ' '.join(labels), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
