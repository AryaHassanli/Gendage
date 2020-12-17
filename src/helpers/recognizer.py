import random

import pandas as pd

from src.helpers.types import AnnotateParameters


class Recognizer:
    def __init__(self,
                 parameters: AnnotateParameters = AnnotateParameters()):
        self.parameters = parameters
        self.total_faces = []
        self.total_unique_faces = 0
        pass

    def recognize(self, encoded_image):
        if self.total_faces:
            dists = [(idx, (encoded_image - emb).norm().item()) for
                     (idx, emb) in self.total_faces]
            dists = pd.DataFrame(dists, columns=['idx', 'dist'])
            means = dists.groupby(by='idx').mean()
            # print(means)
            idx, min_distance = means.idxmin()[0], means.min()[0]

            if min_distance < self.parameters.max_recognize_dist:
                label = idx

                # Add only significantly different faces.
                # TODO define what significant means
                if self.parameters.max_recognize_dist * 2 / 5 < min_distance < \
                        self.parameters.max_recognize_dist * 4 / 5:
                    indices = [i for i, (idx, _) in enumerate(self.total_faces) if idx == label]
                    # Random sub if the number is too high.
                    # TODO substitute the most distant instead of a random
                    if len(indices) > self.parameters.max_img_per_idx \
                            and random.uniform(0, 1) <= self.parameters.substitution_prob:
                        sub_index = random.choice(indices)
                        self.total_faces[sub_index] = (label, encoded_image)
                        pass
                    else:
                        self.total_faces.append((label, encoded_image))

            else:
                label = f'temp_{str(self.total_unique_faces).zfill(3)}'
                self.total_unique_faces += 1
                self.total_faces.append((label, encoded_image))

        else:
            # If there are no identities, store the new one.
            label = f'temp_{str(self.total_unique_faces).zfill(3)}'
            self.total_unique_faces += 1
            self.total_faces.append((label, encoded_image))

        return label

    def clean_total_faces(self):
        # Remove rare faces.
        df = pd.DataFrame(self.total_faces, columns=['idx', 'emb'])
        counts = df['idx'].value_counts()
        rare_idx = counts[counts < self.parameters.min_img_per_idx].index.tolist()
        new_total_faces = [(idx, emb) for (idx, emb) in self.total_faces if idx not in rare_idx]

        # Rename new frequent faces.
        idxes = list(set([idx for idx, _ in new_total_faces]))
        temp_idxes = [idx for idx in idxes if idx.startswith('temp_')]
        start_index = len(idxes) - len(temp_idxes)
        for i, (idx, emb) in enumerate(new_total_faces):
            if idx in temp_idxes:
                idx = f'id_{str(start_index + temp_idxes.index(idx)).zfill(3)}'
            new_total_faces[i] = (idx, emb)

        self.total_faces = new_total_faces
