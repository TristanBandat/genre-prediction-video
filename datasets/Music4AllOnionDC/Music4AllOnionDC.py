"""Music4AllOnionDC dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np

_DESCRIPTION = """
Music4AllOnion DC layer dataset with ResNet vectors.
This dataset has 53 genres based on highest frequency.
Train/Test/Validation = 80% / 10% / 10%
Shape scaled to (4096,).
"""


class Music4AllOnionDC(tfds.core.GeneratorBasedBuilder):
    """
        DatasetBuilder for Music4AllOnionDC dataset.
    """
    VERSION = tfds.core.Version('2.1.0')
    RELEASE_NOTES = {
        '2.1.0': 'Dataset with ResNet vectors and 53 genres.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'input': tfds.features.Tensor(shape=(4096,), dtype=np.float32),
                'label': tfds.features.Tensor(shape=(53,), dtype=np.float32),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('input', 'label'),  # Set to `None` to disable
            # homepage='https://dataset-homepage/',
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = '../../data/'
        paths = [path + 'id_resnet.tsv', path + 'id_genres_binary_short.tsv']
        splits = [0.8, 0.1, 0.1]
        input_df = pd.read_csv(paths[0], sep='\t')
        labels_df = pd.read_csv(paths[1], sep='\t')
        train, test, valid = np.split(input_df, [int(splits[0] * len(input_df)),
                                                 int((splits[0]+splits[1]) * len(input_df))])

        return {
            'train': self._generate_examples(train, labels_df),
            'test': self._generate_examples(test, labels_df),
            'valid': self._generate_examples(valid, labels_df),
        }

    def _generate_examples(self, df, labels_df):
        """Yields examples."""
        for i, line in labels_df.iterrows():
            label = line[2:].to_numpy(dtype=np.float32)

            try:
                data = df[df['id'] == line['id']].iloc[:, 1:].to_numpy(dtype=np.float32)[0]
            except IndexError as e:
                # print(f'Data with index {i} not found.')
                continue

            # # compress the vgg19 data
            # compressed_data = list()
            # for j in range(0, len(vgg19_data), 2):
            #     if j < (len(vgg19_data)/2):
            #         _max = np.max(vgg19_data[j:j + 2])
            #         compressed_data.append(_max)
            #     else:
            #         mean = np.mean(vgg19_data[j:j + 2])
            #         compressed_data.append(mean)
            # vgg19_data = np.asarray(compressed_data).reshape((64, 64))
            # img_data = np.repeat(np.expand_dims(vgg19_data, axis=2), 3, axis=2)

            yield i, {
                'input': data,
                'label': label,
            }
