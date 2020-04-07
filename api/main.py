import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from flask import render_template

#PATH_PREFIX = "gs://topic-sentiment-1/models/"  # for deploy
PATH_PREFIX = "../api-resources/"  # for local
MODEL_PATH = f'{PATH_PREFIX}/model5c'
TOKENIZER_PATH = f'{PATH_PREFIX}/tokenizer.pickle'
MAX_SEQ_LEN = 256

model = None
tokenizer = None
stop_words = None

def get_model():
    global model
    if not model:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def get_tokenizer():
    global tokenier
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def get_stopwords():
    global stop_words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    return stop_words



lookup = {0: 'ajc.com',
          1: 'americanthinker.com',
          2: 'apnews.com',
          3: 'axios.com',
          4: 'bbc.com',
          5: 'boston.com',
          6: 'breitbart.com',
          7: 'cbsnews.com',
          8: 'chicago.suntimes.com',
          9: 'chicagotribune.com',
          10: 'chron.com',
          11: 'cnbc.com',
          12: 'dailykos.com',
          13: 'dallasnews.com',
          14: 'denverpost.com',
          15: 'economist.com',
          16: 'fivethirtyeight.com',
          17: 'forbes.com',
          18: 'foreignpolicy.com',
          19: 'foxnews.com',
          20: 'ft.com',
          21: 'latimes.com',
          22: 'msnbc.com',
          23: 'nbcnews.com',
          24: 'newrepublic.com',
          25: 'newsday.com',
          26: 'newsmax.com',
          27: 'npr.org',
          28: 'nydailynews.com',
          29: 'nypost.com',
          30: 'nytimes.com',
          31: 'prospect.org',
          32: 'reason.com',
          33: 'reuters.com',
          34: 'rt.com',
          35: 'seattletimes.com',
          36: 'slate.com',
          37: 'theatlantic.com',
          38: 'theblaze.com',
          39: 'thehill.com',
          40: 'thenation.com',
          41: 'time.com',
          42: 'utne.com',
          43: 'vox.com',
          44: 'washingtonexaminer.com',
          45: 'washingtonpost.com',
          46: 'wsj.com'}


sites = list(lookup.values())
print(type(sites), sites)
columns = 5
length = len(sites)
items = length // columns + 1

def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, max_seq_length):
    """Token ids from Tokenizer vocab"""
    tokenizer = get_tokenizer()
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def create_single_input(sentence, MAX_LEN):
    global stop_words
    stokens = tokenizer.tokenize(sentence)
    stokens = [token for token in stokens if not token in stop_words]

    stokens = stokens[:MAX_LEN]

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids, masks, segments

model = get_model()
tokenizer = get_tokenizer()
stop_words = get_stopwords()

def get_next_highest(scores):
    print(type(scores), scores)
    max = np.max(scores)
    idx = np.argmax(scores)
    scores[idx] = 0
    return max, idx, scores

def analyze(request):
    results = []
    text = ""
    request_json = request.get_json(silent=True)
    request_args = request.args
    if request_json and 'text' in request_json:
        text = request_json['text']
    elif request_args and 'text' in request_args:
        text = request_args['text']

    print("@@@@@@@", text)

    if text:
        print("text", text)
        ids, masks, segments = create_single_input(text, MAX_SEQ_LEN)
        inputs = [np.asarray([ids]), np.asarray([masks]), np.asarray([segments])]
        model = get_model()
        predictions = model.predict(inputs)
        predictions = predictions[0]
        print("@@@@", predictions)

        for i in range(1, 4):
            score, idx, predictions = get_next_highest(predictions)
            results.append((score * 100, idx, lookup[idx]))

    print("@@@@@", results)
    return render_template('main.html', text=text, sites=sites, columns=5,
                           length=length, items=items, results=results)

