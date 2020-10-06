import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.utils import to_categorical
from google.cloud import logging
from datetime import datetime
import gsutilwrap
import os

BUCKET = 'gs://topic-sentiment-1'  # bucket for data
LOCAL_DATA_DIR = 'data'  # relative name of local directory where input files will be copied
LOCAL_MODEL_DIR = 'model'  # relative name of local directory where final model is saved
LOCAL_CHECKPOINT_DIR = 'checkpoint'  # relative name of local directory where checkpoint files are saved
LOCAL_TEST_OUTPUT_DIR = 'output'  # relative name of local directory where supplementary output is saved
TOKENIZED_DATA_DIR = 'test_tokenized'  # name of bucket subdirectory where input files will be copied from
BUCKET_RESULTS_DIR = 'test-output'  # name of bucket subdirectory where output files will be copied to

#  set up logger
logging_client = logging.Client()
log_name = 'modeling'
logger = logging_client.logger(log_name)


def log_time(msg):
    """
    A short function to add timestamps to logging messages.  Helpful to judge progress when viewing console output
    """
    message = f"@@@@ {datetime.now() : {msg} }"
    print(message)
    logger.log_text(msg)


def get_hdf_from_file(filename, key):
    """
    Reads data from HDF files
    """
    store = pd.HDFStore(f"{filename}")
    data_pdf = store[key]
    store.close()
    return data_pdf


def copy_tokenized_data_local():
    """
    Copies tokens from bucket storage to local directory
    """
    try:
        os.mkdir(LOCAL_DATA_DIR)
    except OSError as err:
        log_time(f"Exception making local data dir: {err}")
    gsutilwrap.copy(pattern=f"{BUCKET}/{TOKENIZED_DATA_DIR}/*", target=f"{LOCAL_DATA_DIR}", recursive=True)


def load_data():
    """
    Loads data used for modeling
    """
    copy_tokenized_data_local()
    data_array = []
    for i in range(0, 10):
        df = get_hdf_from_file(f'{LOCAL_DATA_DIR}/tokens_{i}.h5', 'clean_data')
        df['ids'] = df['ids'].map(lambda x: np.asarray(x, dtype=np.int32))
        df['masks'] = df['masks'].map(lambda x: np.asarray(x, dtype=np.int32))
        df['segments'] = df['segments'].map(lambda x: np.asarray(x, dtype=np.int32))
        data_array.append(df)
    log_time("coalesce data")
    all_data_pdf = pd.concat([data_array[i] for i in range(0, 10)])
    return all_data_pdf


def build_model():
    """
    Loads BERT model for training and adds layer for fine tuning.   We freeze the BERT part of the model and only
    train the last dense layer added to model.  There is a dropout layer as well to help with generalization.
    """
    log_time("Load bert layer")
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=False)
    MAX_SEQ_LEN = 256  # must match what was used in data-wrangling
    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    # Dropout layer to generalize results
    x = tf.keras.layers.Dropout(0.2)(x)
    # fine tuning layer to categorize docuements
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dense_output")(x)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    log_time("compile model")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    model.summary()
    return model


def save_results_to_bucket():
    """
    Saves outputs to a bucket
    """
    gsutilwrap.copy(pattern=f"{LOCAL_CHECKPOINT_DIR}/", target=f"{BUCKET}/{BUCKET_RESULTS_DIR}/", recursive=True)
    gsutilwrap.copy(pattern=f"{LOCAL_MODEL_DIR}/", target=f"{BUCKET}/{BUCKET_RESULTS_DIR}/", recursive=True)
    gsutilwrap.copy(pattern=f"{LOCAL_TEST_OUTPUT_DIR}/", target=f"{BUCKET}/{BUCKET_RESULTS_DIR}/", recursive=True)

log_time("Begin")

log_time(f"TensorFlow Version: {tf.__version__}")
log_time(f"Hub version: {hub.__version__}")

log_time("load data")
all_data_pdf = load_data()

log_time("load domain lookup")
domain_lookup = get_hdf_from_file(f'{LOCAL_DATA_DIR}/domain_lookup.h5', 'domain_lookup')

counts = all_data_pdf.groupby('source_index').source_domain.count().reset_index()
log_time(f"Data count by domain: {counts}")

NUM_CLASSES = len(domain_lookup)
log_time(f"Number of classes:  {NUM_CLASSES}")

all_source_index = all_data_pdf['source_index'].values
all_y_array = to_categorical(all_source_index)

log_time(f"Number of data points: {len(all_y_array)}")

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
inputs = [np.vstack(ids), np.vstack(masks), np.vstack(segments)]

model = build_model()

checkpoint_filepath = LOCAL_CHECKPOINT_DIR + '/epoch-weights.{epoch:02d}'
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

log_time("begin fit")
history = model.fit(inputs,
                    y_train,
                    epochs=7,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[model_checkpoint_callback, early_stopping])

log_time("end fit")
model.save(LOCAL_MODEL_DIR)

log_time("convert inputs for test data")
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

# save data for analysis of predictions
try:
    os.mkdir(LOCAL_TEST_OUTPUT_DIR)
except OSError as err:
    log_time(f"Exception making local output data dir: {err}")
X_test.to_pickle(f'{LOCAL_TEST_OUTPUT_DIR}/X_test.pickle')
pd.DataFrame(y_test).to_pickle(f'{LOCAL_TEST_OUTPUT_DIR}/y_test.pickle')
pd.Series(index_test).to_pickle(f'{LOCAL_TEST_OUTPUT_DIR}/index_test.pickle')

y_comparison['correct'] = (y_comparison.actual == y_comparison.prediction)

overall_acc = y_comparison.correct.mean()
log_time(f"Overall accuracy: {overall_acc}")

pub_averages = y_comparison.groupby('actual').correct.mean()
log_time(f'{pub_averages}')

log_time("Copy output to bucket")
save_results_to_bucket()

log_time("finished")
