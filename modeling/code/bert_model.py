import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
from google.cloud import logging
from datetime import datetime

logging_client = logging.Client()
log_name = 'modeling'
logger = logging_client.logger(log_name)


def log_time(msg):
    """
    A short function to measure execution time of various steps
    """
    print(f"@@@@ {msg} {datetime.now()}")
    logger.log_text(f"@@@@ {msg} {datetime.now()}")


def get_hdf_from_file(filename, key):
    store = pd.HDFStore(f"{filename}")
    data_pdf = store[key]
    store.close()
    return data_pdf

log_time("Begin")

logger.log_text(f"TensorFlow Version: {tf.__version__}")
logger.log_text(f"Hub version: {hub.__version__}")

log_time("load domain lookup")
domain_lookup = get_hdf_from_file('domain_lookup.h5', 'domain_lookup')

log_time("load data")
data_array = []
for i in range(0,10):
    df = get_hdf_from_file(f'tokens_{i}.h5', 'clean_data')
    df['ids'] = df['ids'].map(lambda x: np.asarray(x, dtype=np.int32))
    df['masks'] = df['masks'].map(lambda x: np.asarray(x, dtype=np.int32))
    df['segments'] = df['segments'].map(lambda x: np.asarray(x, dtype=np.int32))
    data_array.append(df)

log_time("coalesce data")
all_data_pdf = pd.concat([data_array[i] for i in range(0, 10)])

del data_array

# inputs = [all_data_pdf['ids'].to_list(), all_data_pdf['masks'].to_list(),
#           all_data_pdf['segments'].to_list()]

counts = all_data_pdf.groupby('source_index').source_domain.count().reset_index()
print(counts)

NUM_CLASSES = len(domain_lookup)
logger.log_text(f"Number of classes:  {NUM_CLASSES}")

all_source_index = all_data_pdf['source_index'].values
all_y_array = to_categorical(all_source_index)

logger.log_text(f"Number of data points: {len(all_y_array)}")

log_time("Split data")

X_train, X_test,  y_train, y_test, index_train, index_test = \
    train_test_split(all_data_pdf,
                     all_y_array,
                     all_source_index,
                     test_size=0.2,
                     shuffle=True,
                     stratify=all_source_index)

ids = X_train['ids'].values
masks = X_train['masks'].values
segments = X_train['segments'].values

log_time("convert inputs")
inputs = [np.vstack(ids), np.vstack(masks), np.vstack(segments)]

log_time("Load bert layer")
bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                          trainable=False)

MAX_SEQ_LEN = 256

input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                    name="segment_ids")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_output")(x)

model = tf.keras.models.Model(
      inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

checkpoint_filepath = 'checkpoints/epoch-weights.{epoch:02d}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='accuracy',
    save_best_only=False,
    mode='auto')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0,
    patience=1,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

log_time("compile model")
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

model.summary()

log_time("begin fit")
history = model.fit(inputs,
                    y_train,
                    epochs=5,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[model_checkpoint_callback, early_stopping])

log_time("end fit")
model.save('model')

log_time("convert inputs test")
ids_test = X_test['ids'].values
masks_test = X_test['masks'].values
segments_test = X_test['segments'].values

inputs_test = [np.vstack(ids_test), np.vstack(masks_test), np.vstack(segments_test)]

log_time("predictions")
y_preds = model.predict(inputs_test)
log_time("getting top choice")
y_top_preds = np.argmax(y_preds, axis=1)



y_comparison = pd.DataFrame(y_top_preds, index_test).reset_index()
y_comparison.columns = ['prediction', 'actual']


X_test.to_pickle('X_test.pickle')
pd.DataFrame(y_test).to_pickle('y_test.pickle')
pd.Series(index_test).to_pickle('index_test.pickle')


y_comparison['correct'] = (y_comparison.actual == y_comparison.prediction)

overall_acc = y_comparison.correct.mean()
logger.log_text(f"Overall accuracy: {overall_acc}")

pub_averages = y_comparison.groupby('actual').correct.mean()
print(pub_averages)

# i = 0
# domains = {}
# source_domains = list(source_domains)
# source_domains.sort()
# for val in source_domains:
#     domains[val] = i
#     i = i + 1
#
# def source_index(source):
#     return domains.value[source]
#
#
#
#
# site_names = pd.Series(domain_lookup.keys(), index=domain_lookup.values())
#
# site_acc = pd.concat([site_names, pub_averages], axis=1)
# site_acc.columns = ['source_domain', 'percentage']
#
# site_count = all_data_pdf.groupby('source_domain').source_index.count()
# site_count.rename(columns={'source_index': 'train_count'}, inplace=True)
#
# site_acc.join(site_count, on="source_domain")
# site_acc = site_acc.sort_values('percentage')
# pd.set_option('display.max_colwidth', None)
# site_acc.head(100)

log_time("finished")
