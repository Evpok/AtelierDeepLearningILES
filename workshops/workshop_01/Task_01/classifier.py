import sys

import numpy as np

from keras.layers import Dense, Activation
from keras.models import Sequential

words2vec = {}
with open('word2vecData_embeddings_dim100.txt') as w2vec_stream:
    for line in w2vec_stream:
        word, *vec = line.strip().split(' ')
        words2vec[word] = np.fromiter(map(float, vec), dtype=np.float)

UNK_WORD_EMBEDDING = np.zeros(100)

labels2vec = {'O': np.array([1, 0]), 'Disease': np.array([0, 1])}


def make_sample(word, label):
    embedding = words2vec.get(word, UNK_WORD_EMBEDDING)
    target = labels2vec[label]
    return embedding, target


def get_data(path):
    entry, target = [], []
    with open(path) as train_stream:
        for line in train_stream:
            try:
                e, t = make_sample(*line.strip().split('\t'))
            except TypeError:
                continue
            entry.append(e)
            target.append(t)
    return np.stack(entry), np.stack(target)


model = Sequential()

model.add(Dense(units=128, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

if len(sys.argv) < 2:
    model.fit(*get_data('train.txt'), epochs=5)
    model.save_weights('model.hdf5')
else:
    model.load_weights(sys.argv[1])

test_w, test_e = next(iter(words2vec.items()))
test = np.expand_dims(test_e, 0)
out = model.predict(test, batch_size=1, verbose=1)
print(test_w, out)

print(model.evaluate(*get_data('test.txt')))
