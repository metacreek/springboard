import numpy as np
from flask import render_template
#from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from google.cloud import storage
#from googleapiclient import errors
import tokenizer as tok
import pandas as pd
import os

# This code uses the following environment variables:
#   GOOGLE_APPLICATION_CREDENTIALS: file location of credentials for running on development
#   DOMAIN_LOOKUP_PATH: path to domain_lookup.h5 file to use

MAX_SEQ_LEN = 256  # This must be same value used in wrangling.  This is the number of tokens used in document analysis.
columns = 5  # Number of columns to use in site dropdown in frontend


def lookup():
    """
    Returns a dictionary that contains numerical keys for sitenames.
    """
    if 'site_lookup' not in globals():
        global site_lookup
        reverse_lookup = setup_lookup()
        dict_items = reverse_lookup.to_dict().items()
        site_lookup = {value: key for (key, value) in dict_items}
    return site_lookup


def setup_lookup():
    """
    Returns the site lookup data from the domain lookup file created during data wrangling.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket('topic-sentiment-1')
    blob = bucket.blob(f'{lookup_path()}/domain_lookup.h5')
    blob.download_to_filename('/tmp/domain_lookup.h5')
    store = pd.HDFStore('/tmp/domain_lookup.h5')
    reverse_lookup = store['domain_lookup']
    store.close()
    return reverse_lookup


def lookup_path():
    """
    Returns the lookup path for the domain_lookup.h5 file, from an env variable or a default value if not found
    """
    return os.environ.get('DOMAIN_LOOKUP_PATH', 'prod1')



def sites():
    """
    Returns a list of all sites used in model
    """
    return list(lookup().values())


number_of_sites = len(sites())
items_per_column = number_of_sites // columns + 1


def get_tokenizer():
    """
    Returns tokenizer.  This must be the same tokenizer as used during wrangling, so the vocab.txt file
    and do_lower_case flag must be the same used during wrangling.   I have extracted the Tokenizer class
    file from TensorFlow so that the code does not need the heavy machinery of TensorFlow to run
    """
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = tok.FullTokenizer(vocab_file='vocab.txt', do_lower_case=True)
    return tokenizer


def get_stopwords():
    """
    Returns a string of stopwords which are removed during tokenization.
    """
    if 'stopwords' not in globals():
        global stopwords
        with open('stopwords', 'r') as f:
            stopwords = f.read()
    return stopwords


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
    tok = get_tokenizer()
    token_ids = tok.convert_tokens_to_ids(tokens, )
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def create_single_input(sentence, MAX_LEN):
    tok = get_tokenizer()
    stokens = tok.tokenize(sentence)
    stokens = [token for token in stokens if token not in get_stopwords()]
    stokens = stokens[:MAX_LEN]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    ids = get_ids(stokens, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids, masks, segments


def get_next_highest(scores):
    """
    returns the max value, the index of the max value, and the rest of the predictions after the max value is zeroed out

    :param scores: prediction data structure returned by prediction service
    """
    max_score = np.max(scores)
    idx = np.argmax(scores)
    scores[idx] = 0
    return max_score, idx, scores


def call_prediction_service(ids, masks, segments):
    """
    Calls the google api prediction service to get document predictions

    :param ids: ids determined during tokenization
    :param masks: masks determined during tokenization
    :param segments: segments determined during tokenization
    """
    service = discovery.build('ml', 'v1')
    name = 'projects/topic-sentiment-269614/models/springboard_capstone_project'
    body = {
        "instances": [{
            "input_word_ids": ids,
            "input_mask": masks,
            "segment_ids": segments}]
    }
    response = service.projects().predict(
        name=name, body=body
    ).execute()
    return response


def analyze(request):
    """
    Main function used to present UI and process predictions

    :param request: flask request object
    """
    results = []
    text = ""
    request_json = request.get_json(silent=True)
    request_args = request.args
    if request_json and 'text' in request_json:
        text = request_json['text']
    elif request_args and 'text' in request_args:
        text = request_args['text']

    if text:
        ids, masks, segments = create_single_input(text, MAX_SEQ_LEN-2)

        response = call_prediction_service(ids, masks, segments)

        if 'error' in response:
            raise RuntimeError(response['error'])

        predictions = response['predictions']
        predictions = predictions[0]['dense_output']

        for i in range(1, 4):
            score, idx, predictions = get_next_highest(predictions)
            results.append((score * 100, idx, lookup()[idx]))

    return render_template('main.html', text=text, sites=sites(), columns=5,
                           length=number_of_sites, items=items_per_column, results=results)

