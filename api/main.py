import numpy as np
from flask import render_template
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
import json

try:
    from . import tokenizer as tok
except:
    import tokenizer as tok

MAX_SEQ_LEN = 256

tokenator = None
stop_words = None

def get_tokenizer():
    global tokenator
    tokenator = tok.FullTokenizer(vocab_file='vocab.txt', do_lower_case=True)
    return tokenator


def get_stopwords():
    global stop_words
    with open('stopwords', 'r') as f:
        stop_words = f.read()
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
    tokenator = get_tokenizer()
    token_ids = tokenator.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def create_single_input(sentence, MAX_LEN):
    global stop_words
    stokens = tokenator.tokenize(sentence)
    stokens = [token for token in stokens if not token in stop_words]

    stokens = stokens[:MAX_LEN]

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids, masks, segments

tokenator = get_tokenizer()
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
        ids, masks, segments = create_single_input(text, MAX_SEQ_LEN-2)
        print("ids", ids)
        print("masks", masks)
        print("segments", segments)
        inputs = [np.asarray([ids]), np.asarray([masks]), np.asarray([segments])]
        print("inputs", inputs)


        service = discovery.build('ml', 'v1')
        name = 'projects/topic-sentiment-269614/models/springboard_capstone_project'
        body = {
            "instances": [{
            "input_word_ids": ids,
            "input_mask": masks,
            "segment_ids": segments}]
        }

        # with open('input.json', 'r') as f:
        #     body = json.load(f)

        print('body', json.dumps(body))

        response = service.projects().predict(
            name=name, body=body
        ).execute()
        print(response)

        if 'error' in response:
            raise RuntimeError(response['error'])

        predictions = response['predictions']
        print("@@@@", predictions)
        predictions = predictions[0]['dense_output']

        for i in range(1, 4):
            score, idx, predictions = get_next_highest(predictions)
            results.append((score * 100, idx, lookup[idx]))

    print("@@@@@", results)
    return render_template('main.html', text=text, sites=sites, columns=5,
                           length=length, items=items, results=results)
