import os
import sys

import cv2
import numpy
from PIL import Image

from src.helpers import eval, faceProcess, getClassifier, getEncoder, logger
from src.helpers.config import config

log = logger.Run()

# TODO: using log
# TODO: Online support
# TODO: Webcam Support

image_extensions = [
    '.jpeg',
    '.jpg',
    '.png'
]
video_extensions = [
    '.mp4',
    '.avi'
]


def main():
    path = config.encoder_pretrain
    encoder = getEncoder.get(config.encoder, pretrained=path).to(config.device)
    encoder.eval()

    classifiers = []
    for i, feature in enumerate(config.features):
        classifiers.append(getClassifier.get(config.classifiers[i], pretrained=config.classifier_pretrain[i],
                                             inputSize=512, numOfClasses=config.num_classes[i]).to(config.device))

    if not os.path.exists(config.input_file):
        print("The input file not exists")
        exit()

    output_path = os.path.join(config.abs_output_dir, "labeled_" + os.path.basename(config.input_file))

    _, file_extension = os.path.splitext(config.input_file)

    if file_extension in image_extensions:
        frame = Image.open(config.input_file)
        frame = numpy.array(frame)
        faces = faceProcess.detect(frame)
        if faces:
            for i, face in enumerate(faces):
                faces[i]['image'] = faceProcess.align(face['image'])
            faces = [face for face in faces if face['image'] is not None]

            for face in faces:
                face['labels'] += label_generator(face['image'], encoder, config.features, classifiers)
                print("Detected: A {} years old {}".format(face['labels'][1], face['labels'][0]))

            print("")
            for face in faces:
                put_labels(frame, face)
        frame = Image.fromarray(frame)
        frame.save(output_path)
        print("Output saved on: {}".format(output_path))

    elif file_extension in video_extensions:
        process(online=0,
                labelGenerators=[label_generator],
                inputFile=config.input_file,
                outputFile=output_path,
                detectionFPS=1,
                features=config.features,
                encoder=encoder,
                classifiers=classifiers)
        print("Output saved on: {}".format(output_path))

    else:
        print("The file extension is not supported yet!")


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


def process(online, labelGenerators, inputFile, outputFile, detectionFPS, encoder, classifiers, **kwargs):
    """
    This 'demo' captures the inputFile and detects faces on it. Then, passes each cropped face to labelGenerator
    functions and retrieves list of labels to put over that face on the original frame. The output video file will be
    saved as outputFile.

    Args:
        online: If it's True the result will be also shown in real-time in a cv2 window
        labelGenerators: list of functions to generate labels. each function should receive a cropped face, and return a
            list of labels
        inputFile: path to the input video file
        outputFile: path to the output video file. If the online option is True, no
        detectionFPS: the fps of processing faces. maximum should be same as input video FPS
        device: device to run on

    Examples:
        demo(online=0,
             labelGenerators=[gendage,recognition],
             inputFile='inputSamples/Recording.mp4',
             outputFile='output/result.avi',
             detectionFPS=60,
             device='cpu')
    Returns:
        None
    """
    if not os.path.exists(inputFile):
        sys.exit('The input file not found')

    inputVideo = cv2.VideoCapture(inputFile)

    inputVideoProps = {
        'fps': inputVideo.get(cv2.CAP_PROP_FPS),
        'width': int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frameCount': int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    detectionFPS = min(detectionFPS, inputVideoProps['fps'])
    detectionFPInput = int(inputVideoProps['fps'] / detectionFPS)

    outputVideo = cv2.VideoWriter(outputFile,
                                  cv2.CAP_FFMPEG,
                                  cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                  inputVideoProps['fps'],
                                  (inputVideoProps['width'], inputVideoProps['height']),
                                  True)

    faces = []
    frameNo = 0
    log.programBegin(inputVideoProps['frameCount'])
    ret, frame = inputVideo.read()
    while inputVideo.isOpened() and ret:
        log.frameBegin()
        frameNo += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frameNo % detectionFPInput == 0:
            log.detectBegin()
            faces.clear()

            faces = faceProcess.detect(frame)
            if faces:
                for i, face in enumerate(faces):
                    faces[i]['image'] = faceProcess.align(face['image'])
                faces = [face for face in faces if face['image'] is not None]

                for labelGenerator in labelGenerators:
                    for face in faces:
                        face['labels'] += labelGenerator(face['image'], encoder, config.features, classifiers)

                log.detectEnd(len(faces))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if faces:
            for face in faces:
                put_labels(frame, face)

        outputVideo.write(frame)
        if online:
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = inputVideo.read()
        log.frameEnd(frameNo)

    outputVideo.release()
    inputVideo.release()
    cv2.destroyAllWindows()
    log.programEnd()
    return
