import os

import cv2
import numpy
import torch
from PIL import Image, ImageDraw, ImageFont

from src.helpers.detectors import MTCNNDetector
from src.helpers.evaluators import EncoderMultitask
from src.helpers.recognizer import Recognizer
from src.helpers.types import AnnotateParameters, Features

supported_formats = {
    'image': [
        '.jpg',
        '.jpeg',
        '.png'
    ],
    'video': [
        '.mp4',
        '.avi'
    ]
}


class Annotate:
    def __init__(self,
                 features: Features = Features(),
                 parameters: AnnotateParameters = AnnotateParameters(),
                 device=torch.device('cpu')):
        self.input_file = ''
        self.save_path = ''
        self.output_file = ''
        self.file_type = ''

        self.features = features
        self.parameters = parameters

        self.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.recognizer = Recognizer(parameters=parameters)
        self.evaluate = EncoderMultitask(features=features, device=device)
        self.detect = MTCNNDetector(device=device)

    def annotate(self, input_file, save_path=None):
        if input_file == 'CAMERA':
            self._annotate_camera()
            return

        if not os.path.isfile(input_file):
            print("Input File not Exists")
            exit()
        self.input_file = input_file

        self.save_path = save_path if save_path is not None else os.path.dirname(input_file)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.output_file = os.path.join(save_path, 'labeled_' + os.path.basename(self.input_file))

        self.file_type = _extension_to_type(os.path.splitext(input_file)[1])

        if self.file_type == 'image':
            self._annotate_image()
        elif self.file_type == 'video':
            self._annotate_video()

    def _annotate_image(self):
        frame = Image.open(self.input_file)

        faces = self.detect(frame)

        if len(faces) == 0:
            print(" No Face Found!")
            return

        faces = self._generate_labels(faces)
        self._put_labels(frame, faces)

        frame.save(self.output_file)
        print("Output saved on: {}".format(self.output_file))
        pass

    def _annotate_video(self):
        input_video = cv2.VideoCapture(self.input_file)
        output_video = cv2.VideoWriter(self.output_file,
                                       cv2.CAP_FFMPEG,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       input_video.get(cv2.CAP_PROP_FPS),
                                       (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                       True)

        frame_count = 0
        total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        faces = []
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            frame_count += 1
            print('\r Frame {}/{} is processing'.format(frame_count, total_frames), end='')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if frame_count % self.parameters.process_every == 0:
                faces = self.detect(frame)
                faces = self._generate_labels(faces)

            if frame_count % self.parameters.clean_every == 0:
                self.recognizer.clean_total_faces()

            self._put_labels(frame, faces)

            frame = numpy.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_video.write(frame)
            # cv2.imshow('out', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        output_video.release()
        input_video.release()
        pass

    def _annotate_camera(self):
        input_video = cv2.VideoCapture(0)

        faces = []
        frame_count = 0
        while input_video.isOpened():
            ret, frame = input_video.read()
            if not ret:
                break
            frame_count += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if frame_count % self.parameters.process_every == 0:
                faces = self.detect(frame)
                faces = self._generate_labels(faces)

            if frame_count % self.parameters.clean_every == 0:
                self.recognizer.clean_total_faces()

            self._put_labels(frame, faces)

            frame = numpy.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('out', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        input_video.release()
        pass

    def _generate_labels(self, faces):
        if len(faces) == 0:
            return faces
        output = self.evaluate(faces)

        if self.features.recognition:
            for i, face in enumerate(faces):
                face.labels['id'] = self.recognizer.recognize(output['encoded'][i])

        if self.features.age:
            for i, face in enumerate(faces):
                age = int(output['ages'][i]) - 3
                age = round(age / 5) * 5
                face.labels['age'] = str(age - 5) + "-" + str(age + 5)

        if self.features.gender:
            for i, face in enumerate(faces):
                face.labels['gender'] = 'Male' if output['genders'][i] < 0.5 else 'Female'

        return faces

    def _put_labels(self, frame, faces):
        if len(faces) == 0:
            return
        frame_width, frame_height = frame.size
        font = ImageFont.truetype("src/helpers/Ubuntu-R.ttf", max(int(frame_height / 24), 15))
        for face in faces:
            x1, y1, x2, y2 = face.box
            x1 = max(x1, 0)
            x2 = min(x2, frame_width - 2)

            y1 = max(y1, font.size)
            y2 = min(y2, frame_height - font.size)

            draw = ImageDraw.Draw(frame)
            if self.features.recognition:
                if not face.labels['id'].startswith('temp'):
                    text = face.labels['id']
                    text_width, text_height = draw.textsize(text, font)
                    draw.text(
                        (
                            (x1 + x2 - text_width) / 2,
                            y1 - font.size
                        ),
                        text, (255, 255, 255), font, stroke_width=2, stroke_fill=(0, 0, 0))

            text = []
            if self.features.gender:
                text.append(face.labels['gender'])

            if self.features.age:
                text.append(face.labels['age'])

            text = ', '.join(text)
            text_width, text_height = draw.textsize(text, font)
            draw.text(
                (
                    (x1 + x2 - text_width) / 2,
                    y2
                ),
                text, (255, 255, 255), font, stroke_width=2, stroke_fill=(0, 0, 0))

            draw.rectangle((x1 - 1, y1 - 1, x2 + 1, y2 + 1),
                           outline=(0, 0, 0), width=1)
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 255, 255), width=1)


def _extension_to_type(extension):
    for file_type in supported_formats:
        if extension in supported_formats[file_type]:
            return file_type
