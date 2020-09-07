import numpy as np
from flask import render_template
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
import json
import tokenizer as tok
import pandas as pd



def get_tokenizer():
    global tokenator
    tokenator = tok.FullTokenizer(vocab_file='vocab.txt', do_lower_case=True)
    return tokenator


def get_stopwords():
    if 'stopwords' not in globals():
        global stopwords
        print("@@@@@@@@@@@@@@@@@@@ Reading stopwords @@@@@@@@@@@@@@@@@@@")
        with open('stopwords', 'r') as f:
            stopwords = f.read()
    return stopwords


def lookup():
    store = pd.HDFStore('domain_lookup.h5')
    reverse_lookup = store['domain_lookup']
    store.close()
    dict_items = reverse_lookup.to_dict().items()
    lookup = {value: key for (key, value) in dict_items}
    return lookup

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
    stokens = tokenator.tokenize(sentence)
    stokens = [token for token in stokens if not token in get_stopwords()]

    stokens = stokens[:MAX_LEN]

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids, masks, segments


def sites():
    return list(lookup().values())

MAX_SEQ_LEN = 256

tokenator = None

columns = 5
length = len(sites())
items_per_column = length // columns + 1

tokenator = get_tokenizer()

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
            results.append((score * 100, idx, lookup()[idx]))

    print("@@@@@", results)
    return render_template('main.html', text=text, sites=sites(), columns=5,
                           length=length, items=items_per_column, results=results)
