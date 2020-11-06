import os
import sys
import time

import cv2
import numpy as np
from facenet_pytorch import MTCNN

from config import config

mtcnn = MTCNN(keep_all=True, device=config.device)


def demo(online, labelGenerators, inputFile, outputFile, detectionFPS, device):
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
    __log = __Log(inputVideoProps['frameCount'])
    ret, frame = inputVideo.read()
    while inputVideo.isOpened() and ret:
        __log.frameBegin()
        frameNo += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frameNo % detectionFPInput == 0:
            __log.detectBegin()
            faces.clear()

            faces = __detect(frame, device)
            if faces:
                for i, face in enumerate(faces):
                    faces[i]['image'] = __processFace(face['image'], device)
                faces = [face for face in faces if face['image'] is not None]

                for labelGenerator in labelGenerators:
                    for face in faces:
                        face['labels'] += labelGenerator(face['image'])

                __log.detectEnd(len(faces))

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
        __log.frameEnd(frameNo)

    outputVideo.release()
    inputVideo.release()
    cv2.destroyAllWindows()
    __log.programEnd()
    return


def __detect(frame, device):
    global mtcnn
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
            'image': frame[y1:y2, x1:x2]
        })
    return faces


def __processFace(faceImage, device):
    return faceImage


def __putLabels(frame, face):
    x1, y1, x2, y2 = face['box']
    labels = face['labels']
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.putText(frame, ' '.join(labels), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)


class __Log:
    def __init__(self, frameCount):
        self.__frameCount = frameCount

        self.__frameStart = 0
        self.__frameTime = 0
        self.__totalFrameTime = 0

        self.__detectStart = 0
        self.__detectTime = 0
        self.__totalDetectTime = 0
        self.__totalDetections = 0

    def frameBegin(self):
        self.__frameStart = time.time()
        pass

    def frameEnd(self, frameNo):
        self.__frameTime = time.time() - self.__frameStart
        self.__totalFrameTime += self.__frameTime
        print('Frame {}/{} Processed in {}ms\t'.format(
            frameNo,
            self.__frameCount,
            round(1000 * self.__frameTime, 1)
        ), end='\n')
        pass

    def detectBegin(self):
        self.__detectStart = time.time()
        pass

    def detectEnd(self, numOfFaces):
        self.__detectTime = time.time() - self.__detectStart
        self.__totalDetectTime += self.__detectTime
        self.__totalDetections += numOfFaces
        pass

    def programBegin(self):
        pass

    def programEnd(self):
        print("\n")
        print("Total Frames:", self.__frameCount)
        print("Total Frame Process Time: {}ms, Average: {}ms".format(
            round(1000 * self.__totalFrameTime, 1),
            round(1000 * self.__totalFrameTime / self.__frameCount, 1)
        ))
        print()
        print("Total Faces detected:", self.__totalDetections)
        print("Total Detection Time: {}ms, Average: {}ms".format(
            round(1000 * self.__totalDetectTime, 1),
            round(1000 * self.__totalDetectTime / self.__totalDetections, 1)
        ))
        pass
