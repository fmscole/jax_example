import tensorflow_datasets as tfds
ds = tfds.load('./mnist', split='train', shuffle_files=True)