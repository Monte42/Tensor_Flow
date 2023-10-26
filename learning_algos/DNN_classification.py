from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES = ['Setosa', 'Versicolor','Virginica']

train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# print(train.head())
# print(train.shape)

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

feature_columns = []
for k in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=k))
# print(feature_columns)

classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[30,10],
        n_classes=3
    )

classifier.train( # Crashes here
        input_fn = lambda: input_fn(train,train_y, training=True),
        steps=5000
    )