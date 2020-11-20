import os
import sys

import cv2
import torch

from config import config
from helpers import eval
from helpers import faceProcess
from helpers import getNet
from helpers import logger
from helpers import parseArguments

args = parseArguments.parse('demo')
config.setup(args)

log = logger.Demo()
log.environment()


def main():
    process(online=0,
            labelGenerators=[gendage],
            inputFile='D:/MsThesis/inputSamples/Harry.mp4',
            outputFile='output/result.avi',
            detectionFPS=5,
            features=['age', 'gender'])


model = getNet.get('resnet18Multi').to(config.device)
model.load_state_dict(
    torch.load('D:/Gendage/output/modelDouble.pt')
)

class MovingAverage:
    def __init__(self, window):
        self.list = []
        self.window = window

    def addItem(self, item):
        self.list.append(item)
        if len(self.list) > self.window:
            self.list.pop(0)
        return sum(self.list) / len(self.list)


movingAvgs = {}


def gendage(face, features=None):
    if len(movingAvgs) == 0:
        for feature in features:
            movingAvgs[feature] = MovingAverage(7)
    out = eval.faceImage(face, model, features=features)
    outputList = []
    for feature in features:
        outputList.append(str(round(
            movingAvgs[feature].addItem(out[feature]),
            2)))
    return outputList


def process(online, labelGenerators, inputFile, outputFile, detectionFPS, **kwargs):
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
                        face['labels'] += labelGenerator(face['image'], **kwargs)

                log.detectEnd(len(faces))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if faces:
            for face in faces:
                __putLabels(frame, face)

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


def __putLabels(frame, face):
    x1, y1, x2, y2 = face['box']
    labels = face['labels']
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.putText(frame, ' '.join(labels), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)


if __name__ == '__main__':
    main()
