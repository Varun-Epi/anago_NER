import anago
import codecs
import numpy as np

from os.path import dirname, abspath, join
from anago.utils import load_data_and_labels


def load_glove(file):
    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            model[word] = vector

    return model


if __name__ == '__main__':
    DATA_ROOT = join(dirname(abspath(__file__)), 'data/cypher')
    EMBEDDING_PATH = '/home/varun/Downloads/glove.6B/glove.6B.100d.txt'

    train_path = join(DATA_ROOT, 'train.txt')
    valid_path = join(DATA_ROOT, 'valid.txt')

    print(train_path)

    print('Loading data...')
    x_train, y_train = load_data_and_labels(train_path)
    x_valid, y_valid = load_data_and_labels(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')

    embeddings = load_glove(EMBEDDING_PATH)

    # Use pre-trained word embeddings
    model = anago.Sequence(embeddings=embeddings)
    model.fit(x_train, y_train, x_valid, y_valid, epochs=10)
