from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, date_format, rank
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
import tensorflow_hub as hub
import bert
import pandas as pd
from google.cloud import storage
from google.cloud import logging
import sys

def setup_logging():
    logging_client = logging.Client()
    log_name = 'wrangling'
    return logging_client.logger(log_name)

spark = (SparkSession.builder
         .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

logger = setup_logging()
sc = spark.sparkContext

logger.log_text(f"@@@@ Variables: {dir()}")
logger.log_text(f"@@@@ argv: {sys.argv}")


def log_time(msg):
    """
    A short function to measure execution time of various steps
    """
    logger.log_text(f"@@@@ {msg} {datetime.now()}")


RAW_DATA = sys.argv[1]
TOKENIZED_DATA_DIR = sys.argv[2]

log_time("Begin processing")

raw_df = spark.read.json(RAW_DATA)


def drop_columns(sdf):
    # delete columns we don't need
    columns_to_drop = ['filename', 'image_url', 'localpath', 'title_page', 'title_rss']
    sdf = sdf.drop(*columns_to_drop)
    return sdf


clean_df = drop_columns(raw_df)


def add_published(sdf):
    # create timestamp version of string date_publish
    sdf = sdf.withColumn("published", (col("date_publish").cast("timestamp")))
    # Useful in cleaning and analysis
    sdf = sdf.withColumn("year", (date_format(col("published"), 'yyyy').cast("int")))
    sdf = sdf.withColumn("month", (date_format(col("published"), 'M').cast("int")))
    return sdf


clean_df = add_published(clean_df)


def handle_language(sdf):
    # We create a new column text_or_desc that will be used in analysis.
    # This uses the text column data, if present, and falls back to description if the text column was empty.
    sdf = sdf.withColumn("text_or_desc",
                         expr("case when text IS NULL THEN description ELSE text END"))
    # we assume that the language is English (en) if the language was not specified.
    # We are excluding bbc.com where this was not a good assumption.
    # See EDA-inital notebook for more info.
    sdf = sdf.withColumn("language_guess",
                         expr(
                             "case when (language IS NULL AND source_domain NOT IN ('bbc.com')) THEN 'en' ELSE language END"))
    # We are only doing english
    sdf = sdf.where("language_guess = 'en'")
    return sdf


clean_df = handle_language(clean_df)

# remove articles with a particular repeated statement.  One site (thenation.com) had a particlar habit
# of including long sales blurbs in the story text.  Because there were multiple versions of this, I decided
# to not use any such news story, because it would give the classifier an unfair signal of the source domain.
clean_data_sdf = clean_df.filter(~clean_df.text_or_desc.contains("Sign up for Take Action"))


def drop_empty(sdf):
    # get rid of any rows where the date of publish, the title or the text/description is empty.
    sdf = sdf.na.drop(subset=['date_publish', 'published', 'text_or_desc', 'title'])
    return sdf


clean_df = drop_empty(clean_df)


def remove_duplicates(sdf):
    # drop duplicates
    sdf = sdf.dropDuplicates(['url', 'date_publish'])
    # additional duplicate removal
    sdf = sdf.dropDuplicates(['text_or_desc'])
    return sdf


clean_df = remove_duplicates(clean_df)


def year_filter(sdf):
    # Use only data since 2010
    sdf = sdf.where('year >= 2010')
    return sdf


clean_df = year_filter(clean_df)

log_time("Begin leveling data")


def level_data(sdf):


    subtotal = sdf.count()
    # threshold is based on previous analysis of data
    upper_threshold = 20000
    logger.log_text(f"Upper threshold is {upper_threshold}")
    # see https://stackoverflow.com/a/38398563/914544
    window = Window.partitionBy(sdf['source_domain']).orderBy(sdf['published'].desc())
    sdf = sdf.select('*', rank().over(window).alias('rank')).filter(col('rank') <= upper_threshold)
    # Get rid of publications with small data
    total_by_publication = sdf.groupby('source_domain').count()
    vals = total_by_publication.select('source_domain').collect()
    all_publications = [val.source_domain for val in vals]
    logger.log_text(f"all publications {all_publications}")
    total_by_publication = total_by_publication.where(f'count == {upper_threshold}')
    vals = total_by_publication.select('source_domain').collect()
    keep_publications = [val.source_domain for val in vals]
    logger.log_text(f"keep publications {keep_publications}")
    sdf = sdf.where(col('source_domain').isin(keep_publications))
    return sdf


clean_df = level_data(clean_df)
log_time("Begin tokenizing")
MAX_SEQ_LEN = 256
MAX_SEQ_LEN_BC = sc.broadcast(MAX_SEQ_LEN)

# This project seeks to classify the origin of news stories.  Some publishers
# include the name of the publication in the crawled stories.  I felt this was a
# "cheat" that made it too easy to classify stories, so I removed the name
# from the story body.  The SUBSTITUTION dictionary lists these names to be
# removed in later code.  In addition, there were certain words that were found
# to be extremely frequent for certain publications.  This was found during
# NER (named-entity recognition) which was performed as part of an approach
# to classifying stories.  That approach was eventually abandoned but still
# played a useful part in analysis.  It can be seen on the other-approaches
# code branch.   Additionally, the site host name is removed from the story text.
SUBSTITUTION = {'ajc.com': ['Atlanta Journal-Constitution', 'Subscribe'],
                'americanthinker.com': ['Full Post'],
                'apnews.com': ['AP', 'Associated Press', 'Business Wire', 'Copyright', 'ResearchAndMarkets.com'],
                'axios.com': ['Axios'],
                'bbc.com': ['BBC'],
                'boston.com': ['Boston Globe', 'Copyright'],
                'breitbart.com': ['Breitbart News', 'Breitbart'],
                'cbsnews.com': ['CBS News', 'CBS'],
                'chicago.suntimes.com': ['Sun-Times'],
                'chicagotribune.com': [],
                'chron.com': ['CHRON', 'PHOTO'],
                'cnbc.com': ['CNBC'],
                'dailykos.com': ['Daily Kos'],
                'dallasnews.com': ['unitName', 'dfpPosition'],
                'denverpost.com': ['Denver Post'],
                'economist.com': ['Daily Dispatch'],
                'fivethirtyeight.com': ['FiveThirtyEight'],
                'forbes.com': ['Forbes'],
                'foreignpolicy.com': ['Foreign Policy'],
                'foxnews.com': ['Fox News', 'FOX NEWS APP'],
                'ft.com': ['Financial Times'],
                'latimes.com': ['Los Angeles Times'],
                'msnbc.com': ['MSNBC'],
                'nbcnews.com': ['NBC News', 'NBC'],
                'newrepublic.com': ['The New Republic', 'New Republic'],
                'newsday.com': ['Newsday', 'Subscribe'],
                'newsmax.com': ['Newsmax'],
                'npr.org': ['NPR', 'Enlarge', 'Copyright'],
                'nydailynews.com': ['Daily News'],
                'nypost.com': ['New York Post'],
                'nytimes.com': ['New York Times', 'Archives', 'Buy Reprints', 'Subscribe'],
                'prospect.org': ['American Prospect', 'Prospect'],
                'reason.com': ['Reason'],
                'reuters.com': ['Reuters', 'REUTERS', 'PHOTO'],
                'rt.com': ['RT', 'PHOTO'],
                'seattletimes.com': ['Seattle Times'],
                'slate.com': ['Slate'],
                'theatlantic.com': ['The Atlantic'],
                'theblaze.com': ['TheBlaze', 'The Blaze', 'Blaze'],
                'thehill.com': ['The Hill', 'Capitol Hill Publishing Corp'],
                'thenation.com': ['The Nation', 'Subscribe', 'Current Issue View'],
                'time.com': ['rzrzrz'],
                'utne.com': ['Utne Reader', 'Utne'],
                'vox.com': ['Vox', 'Subscribe'],
                'washingtonexaminer.com': ['Washington Examiner', 'Examiner'],
                'washingtonpost.com': ['Washington Post'],
                'wsj.com': ['Wall Street Journal', 'Copyright']}

# make a regex pattern for removal
for key in SUBSTITUTION:
    SUBSTITUTION[key] = "(" + "|".join(SUBSTITUTION[key]) + "|" + key + "|\n)"

# broadcast data to workers
SUBSTITUTION_BC = sc.broadcast(SUBSTITUTION)


def add_regex(source):
    """
    Returns the regex string to remove for a particular website.

    :param source: string, canonical website domain name
    """
    return SUBSTITUTION_BC.value[source]


udf_add_regex = F.udf(add_regex)


def clean_text(arr):
    """
    Removes unwanted text from news stories.

    :param arr: array of strings. The first element is the text to be processed.  The second element is the regex pattern to apply to the text.

    Using an array for input, rather than two separate arguments is a trick that is used to make it easier to create a UDF (user-defined function) that can be
    applied in parallel across workers.
    """
    text = arr[0]
    pattern = arr[1]
    return re.sub(pattern, ' ', text)


udf_clean_text = F.udf(clean_text)


def get_tokens(text):
    """
    Returns correctly formatted tokens for given text.  This also removes stopwords.

    :param text: string, body of news story
    """
    tokens = tokenizer.tokenize(text)
    stop_words_val = stop_words_bc.value
    tokens = [token for token in tokens if token not in stop_words_val]
    token_length = MAX_SEQ_LEN_BC.value - 2
    tokens = ["[CLS]"] + tokens[:token_length] + ["[SEP]"]
    return tokens


udf_get_tokens = F.udf(get_tokens)


def get_masks(tokens):
    """
    Returns a list of masks for the tokens.  An individual mask is 1 if the token is text and 0 if it is empty

    :param tokens: list of tokens
    """
    max_seq_length = MAX_SEQ_LEN_BC.value
    masks = [1] * len(tokens) + [0] * (max_seq_length - len(tokens))
    return masks


def get_segments(tokens):
    """
    Returns a list of segments, where an individual segment is 0 for the first sequence, and one for the second, if present

    :param tokens: list of tokens
    """
    max_seq_length = MAX_SEQ_LEN_BC.value
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    segments = segments + [0] * (max_seq_length - len(tokens))
    return segments


def get_ids(tokens):
    """
    Returns a list of integer word ids for use with Bert model

    :param tokens: list of tokens
    """
    max_seq_length = MAX_SEQ_LEN_BC.value
    token_ids = tokenizer.convert_tokens_to_ids(tokens, )
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


udf_get_masks = F.udf(get_masks, T.ArrayType(T.IntegerType()))
udf_get_segments = F.udf(get_segments, T.ArrayType(T.IntegerType()))
udf_get_ids = F.udf(get_ids, T.ArrayType(T.IntegerType()))


def get_source_domains(sdf):
    """
    Returns a dict with canonical domain names as keys and unique integer identifiers for the sorted domain namess

    :param sdf: spark dataframe that contains at least one instance of all domain names
    """
    log_time("Begin source domains")
    source_domains = sdf.select(F.collect_list('source_domain')).first()[0]
    source_domains = set(source_domains)
    i = 0
    domains = {}
    source_domains = list(source_domains)
    source_domains.sort()
    for val in source_domains:
        domains[val] = i
        i = i + 1
    return domains


def source_index(source):
    """
    Returns id corresponding to a canonical domain name

    :param source: string, canonical domain name
    """
    return domains_bc.value[source]


udf_source_index = F.udf(source_index)


def process_data(raw_data_sdf, bert_layer):
    """
    Performs the bulk of the work of tokenization and other cleanups. Returns a reduced spark data frame including ids, masks, and segments, and other helpful elements

    :param raw_data_sdf: spark dataframe, the news stories to be processed
    :param bert_layer: tensorflow Keras layer for the BERT model being used.
    """
    global stop_words_bc, tokenizer, domains_bc
    # add weeks column
    clean_data_sdf = raw_data_sdf.withColumn('weeks',
                                               F.floor(F.datediff(F.col('published'), F.lit('2010-01-01')) / 7))
    log_time("Begin regex")

    clean_data_sdf = clean_data_sdf.withColumn('regex', udf_add_regex('source_domain'))
    # remove all the identifying text from stories
    clean_data_sdf = clean_data_sdf.withColumn('clean_text', udf_clean_text(F.array('text_or_desc', 'regex')))

    log_time("Begin tokenizer")
    # load the tokenizer for the BERT model used
    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    clean_data_sdf = clean_data_sdf.withColumn('tokens', udf_get_tokens('clean_text'))

    log_time("Begin masks, etc.")

    clean_data_sdf = clean_data_sdf.withColumn('masks', udf_get_masks('tokens'))
    clean_data_sdf = clean_data_sdf.withColumn('segments', udf_get_segments('tokens'))
    clean_data_sdf = clean_data_sdf.withColumn('ids', udf_get_ids('tokens'))
    clean_data_sdf = clean_data_sdf.withColumn('source_index', udf_source_index('source_domain').cast('int'))
    # let's slim down the dataframe before we save it to disk.
    clean_data_sdf = clean_data_sdf[['source_domain', 'text_or_desc', 'clean_text', 'published_date', 'year',
                                     'title', 'url', 'weeks', 'tokens', 'masks', 'segments', 'ids', 'source_index']]

    log_time("Convert to pandas")
    return clean_data_sdf


log_time('Begin stopwords')
# loads the nltk english stopwords that should be excluded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words_bc = sc.broadcast(stop_words)

log_time("Begin Keras layer")
# if you wanted to use a different BERT model, here is where you would specify it.
# Because we are fine-tuning the BERT model, we "freeze" the BERT model itself with trainable=False
# If we did not do this, the number of trainable parameters would skyrocket and training would take
# a very long time.
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

log_time("Split data")
# The spark dataframe is quite large, around 1.5 million stories, so I decided to
# split it into ten parts.  This allowed me to do analysis with a subset of data and
# it made it easier to understand how long the saving process was taking.
# it also saved on memory required for processing.
raw_data_sdf = clean_df
raw_data_sdf.printSchema()
raw_data_split_sdf = raw_data_sdf.randomSplit([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

gcs = storage.Client()
iteration = 0
for df in raw_data_split_sdf:
    log_time(f"Begin iteration {iteration}")
    if iteration == 0:
        # save the domains with their canonical ids.  This only works with large enough data sets
        # such that it has all domains that will be used.  We only do this the first time through.
        domains = get_source_domains(df)
        domains_bc = sc.broadcast(domains)
        domain_lookup = pd.Series(domains)
        store = pd.HDFStore('domain_lookup.h5')
        store['domain_lookup'] = domain_lookup
        store.info()
        store.close()
        # see https://stackoverflow.com/a/56787083
        gcs.bucket('topic-sentiment-1').blob(f'{TOKENIZED_DATA_DIR}/domain_lookup.h5').upload_from_filename(
            'domain_lookup.h5')

    clean_sdf = process_data(df, bert_layer)
    clean_data_pdf = clean_sdf.toPandas()
    log_time("Begin store")
    filename = f"tokens_{iteration}.h5"
    store = pd.HDFStore(filename)
    store['clean_data'] = clean_data_pdf
    store.info()
    store.close()
    log_time("Begin move to bucket")
    gcs.bucket('topic-sentiment-1').blob(f"{TOKENIZED_DATA_DIR}/{filename}").upload_from_filename(filename)
    iteration = iteration + 1

log_time("Finished")
