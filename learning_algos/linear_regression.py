# PREP
# ===================
from __future__ import absolute_import,division,print_function,unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# LESSON 1
# ===================
# print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# print(dftrain.head())
# print(dftrain.loc[2])
# print(dftrain['age'])
# dftrain['age'].hist(bins=20)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='bar')
# pd.concat([dftrain,y_train], axis=1).groupby('sex').survived.mean().plot(kind='bar').set_xlabel('Survival By Gender')
# plt.show()

# LESSON 2
# ===================
CATEGORICAL_COLUMNS = [
    'sex',
    'n_siblings_spouses',
    'parch',
    'class',
    'deck',
    'embark_town',
    'alone'
]
NUMERIC_COLUMNS = ['age','fare']

feature_columns = []

for fn in CATEGORICAL_COLUMNS:
    vocab = dftrain[fn].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(fn,vocab))

for fn in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(fn,dtype=tf.float32))

# print(feature_columns)

# Lesson 3
# ===================
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
results = linear_est.evaluate(eval_input_fn)

print(results['accuracy'])

result = list(linear_est.predict(eval_input_fn))
print(result[0]['probabilities'])

