import time

import cv2
from facenet_pytorch import MTCNN


def demo(labelGenerator, inputFile, outputFile, detectionFPS, device):
    """
    This 'demo' captures the inputFile and detects faces on it. Then, passes each cropped face to labelGenerator
    function and retrieves list of labels to put over that face on the original frame. The output video file will be
    saved as outputFile.

    Args:
        labelGenerator: function that receives a cropped face, and returns list of labels
        inputFile: path to the input video
        outputFile: path to the output video
        detectionFPS: the fps of processing faces.
        device: device name

    Examples:
        demo(ageEstimator,
             os.path.join(config.baseDir, 'inputSamples/Recording.mp4'),
             os.path.join(config.outputDir, 'result.avi'),
             15,
             'cpu')
    Returns:
        None
    """
    mtcnn = MTCNN(keep_all=True, device=device)
    inputVideo = cv2.VideoCapture(inputFile)
    inputFPS = inputVideo.get(cv2.CAP_PROP_FPS)
    frameWidth = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    outputVideo = cv2.VideoWriter(outputFile,
                                  cv2.CAP_FFMPEG,
                                  cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                  inputFPS,
                                  (frameWidth, frameHeight),
                                  True)
    frameNo = 0
    boxes = []
    labels = []
    ret = True
    maxExecutionTime = 0
    totalExecutionTime = 0
    while inputVideo.isOpened() and ret:
        startTime = time.time()
        ret, frame = inputVideo.read()
        if frameNo % int(inputFPS / detectionFPS) == 0:
            labels = []
            boxes, _ = mtcnn.detect(frame)
            for box in boxes:
                box = box.astype(int)
                x1, y1, x2, y2 = box
                croppedFace = frame[y1:y2, x1:x2]
                labels.append(labelGenerator(croppedFace))
        for i, box in enumerate(boxes):
            box = box.astype(int)
            x1, y1, x2, y2 = box
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.putText(frame, ' '.join(labels[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #cv2.imshow('frame', frame)
        outputVideo.write(frame)
        endTime = time.time()
        executionTime = endTime - startTime
        maxExecutionTime = max(maxExecutionTime, executionTime)
        totalExecutionTime += executionTime
        print(
            '\rFrame {}/{}.\t Execution Time: {}ms\t'.format(frameNo, totalFrames, round(1000 * executionTime, 1)),
            end='')
        frameNo += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('\nTotal Execution Time: {}ms, Max Execution per Frame: {}ms'.format(round(1000 * totalExecutionTime, 1),
                                                                               round(1000 * maxExecutionTime, 1)))
    outputVideo.release()
    inputVideo.release()
    cv2.destroyAllWindows()
    return
