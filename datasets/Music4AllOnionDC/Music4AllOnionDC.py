"""Music4AllOnionDC dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import numpy as np

# TODO(Music4AllOnionDC): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(Music4AllOnionDC): BibTeX citation
_CITATION = """
"""


class Music4AllOnionDC(tfds.core.GeneratorBasedBuilder):
    """
        DatasetBuilder for Music4AllOnionDC dataset.
    """
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'input': tfds.features.Tensor(shape=(4096,), dtype=tf.float32),
                'label': tfds.features.Tensor(shape=(685,), dtype=tf.float32),
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
        path = '../data/'

        return {
            'train': self._generate_examples([path + 'id_incp.tsv', path + 'id_genres_binary.tsv']),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        input_df = pd.read_csv(path[0], sep='\t')
        labels_df = pd.read_csv(path[1], sep='\t')
        for i, line in input_df.iterrows():
            yield i, {
                'input': line[1:].to_numpy(dtype=np.float32),
                'label': labels_df[labels_df['id'] == line['id']].T[2:].T.to_numpy(dtype=np.float32)[0],
            }
