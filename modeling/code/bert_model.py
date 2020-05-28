import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)


def get_hdf_from_file(filename, key):
    store = pd.HDFStore(filename)
    data_pdf = store[key]
    store.close()
    return data_pdf


domain_lookup = get_hdf_from_file('domain_lookup.h5', 'domain_lookup')

tfrecords_filename = 'tokens.tfrecord'



data_array = []
for i in range(0,1):
    df = get_hdf_from_file(f'tokenized_{i}.h5', 'clean_data')
    df['ids'] = df['ids'].map(lambda x: np.asarray(x, dtype=np.int32))
    df['masks'] = df['masks'].map(lambda x: np.asarray(x, dtype=np.int32))
    df['segments'] = df['segments'].map(lambda x: np.asarray(x, dtype=np.int32))
    data_array.append(df)

all_data_pdf = pd.concat([data_array[i] for i in range(0, 1)])

del data_array

# inputs = [all_data_pdf['ids'].to_list(), all_data_pdf['masks'].to_list(),
#           all_data_pdf['segments'].to_list()]

counts = all_data_pdf.groupby('source_index').source_domain.count().reset_index()
print(counts)

NUM_CLASSES = len(domain_lookup)
print("Number of classes: ", NUM_CLASSES)

all_source_index = all_data_pdf['source_index']
all_y_array = to_categorical(all_source_index)

X_train, X_test,  y_train, y_test, index_train, index_test = train_test_split(all_data_pdf, all_y_array,  all_source_index,
      test_size=0.2, shuffle=True, stratify=all_source_index)

ids = X_train['ids'].values
masks = X_train['masks'].values
segments = X_train['segments'].values

inputs = [np.vstack(ids), np.vstack(masks), np.vstack(segments)]

bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)

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

checkpoint_filepath = 'gs://topic-sentiment-1/checkpoints'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_acc',
    mode='max')

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True
)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

model.summary()

history = model.fit(inputs,
                    y_train,
                    epochs=1,
                    batch_size=1000,
                    shuffle=True,
                    callbacks=[model_checkpoint_callback, early_stopping])

y_preds = model.predict(X_test)
y_top_preds = np.argmax(y_preds, axis=1)

model.save('gs://topic-sentiment-1/model-v1')

y_comparison = pd.DataFrame(index_test, y_top_preds).reset_index()
y_comparison.columns = ['prediction', 'actual']

y_comparison['correct'] = (y_comparison.actual == y_comparison.prediction)

overall_acc = y_comparison.correct.mean()
print("Overall accuracy:", overall_acc)

pub_averages = y_comparison.groupby('actual').correct.mean()

site_names = pd.Series(domain_lookup.keys(), index=domain_lookup.values())

site_acc = pd.concat([site_names, pub_averages], axis=1)
site_acc.columns = ['source_domain', 'percentage']

site_count = all_data_pdf.groupby('source_domain').source_index.count()
site_count.rename(columns={'source_index': 'train_count'}, inplace=True)

site_acc.join(site_count, on="source_domain")
site_acc = site_acc.sort_values('percentage')
pd.set_option('display.max_colwidth', None)
site_acc.head(100)


