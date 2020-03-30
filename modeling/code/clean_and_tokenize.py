from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
import tensorflow_hub as hub
import bert
import pandas as pd
from google.cloud import storage

spark = (SparkSession.builder
        .config("spark.debug.maxToStringFields", 100)
         .getOrCreate()
         )

sc = spark.sparkContext
sqlContext = SQLContext(sc)

INPUT_NAME =  'gs://topic-sentiment-1/sample2.json'
OUTPUT_NAME = 'test.h5'


def log_time(msg):
    print(f"@@@@ {msg} {datetime.now()}")


log_time("Begin processing")

raw_data_sdf = spark.read.json(INPUT_NAME)

MAX_SEQ_LEN = 256
MAX_SEQ_LEN_BC = sc.broadcast(MAX_SEQ_LEN)

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

for key in SUBSTITUTION:
    SUBSTITUTION[key] = "(" + "|".join(SUBSTITUTION[key]) + "|" + key + "|\n)"

SUBSTITUTION_BC = sc.broadcast(SUBSTITUTION)

# remove articles with a particular repeated statement
clean_data_sdf = raw_data_sdf.filter(~raw_data_sdf.text_or_desc.contains("Sign up for Take Action"))

# add weeks column
clean_data_sdf = clean_data_sdf.withColumn('published_date', F.to_date(F.col('published')))
clean_data_sdf = clean_data_sdf.withColumn('weeks', F.floor(F.datediff(F.col('published'), F.lit('2010-01-01'))/7))

log_time("Begin regex")

def add_regex(source):
    return SUBSTITUTION_BC.value[source]


udf_add_regex = F.udf(add_regex)

clean_data_sdf = clean_data_sdf.withColumn('regex', udf_add_regex('source_domain'))


def clean_text(arr):
    text = arr[0]
    pattern = arr[1]
    return re.sub(pattern, ' ', text)


udf_clean_text = F.udf(clean_text)

clean_data_sdf = clean_data_sdf.withColumn('clean_text', udf_clean_text(F.array('text_or_desc', 'regex')))

log_time('Begin stopwords')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words_bc = sc.broadcast(stop_words)

log_time("Begin Keras layer")

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)

log_time("Begin tokenizer")

FullTokenizer = bert.bert_tokenization.FullTokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file,do_lower_case)


def get_tokens(text):
    stokens = tokenizer.tokenize(text)
    stop_words = stop_words_bc.value
    stokens = [token for token in stokens if not token in stop_words]
    token_length = MAX_SEQ_LEN_BC.value - 2
    stokens = ["[CLS]"] + stokens[:token_length] + ["[SEP]"]
    return stokens


udf_get_tokens = F.udf(get_tokens)

clean_data_sdf = clean_data_sdf.withColumn('tokens', udf_get_tokens('clean_text'))

log_time("Begin masks, etc.")

def get_masks(tokens):
    max_seq_length = MAX_SEQ_LEN_BC.value
    masks = [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
    return masks


def get_segments(tokens):
    """Segments: 0 for the first sequence, 1 for the second"""
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
    """Token ids from Tokenizer vocab"""
    max_seq_length = MAX_SEQ_LEN_BC.value
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


udf_get_masks = F.udf(get_masks, T.ArrayType(T.IntegerType()))
udf_get_segments = F.udf(get_segments, T.ArrayType(T.IntegerType()))
udf_get_ids = F.udf(get_ids, T.ArrayType(T.IntegerType()))

clean_data_sdf = clean_data_sdf.withColumn('masks', udf_get_masks('tokens'))
clean_data_sdf = clean_data_sdf.withColumn('segments', udf_get_segments('tokens'))
clean_data_sdf = clean_data_sdf.withColumn('ids', udf_get_ids('tokens'))

log_time("Begin source domains")

source_domains = clean_data_sdf.select(F.collect_list('source_domain')).first()[0]
source_domains = set(source_domains)

i = 0
domains = {}
source_domains = list(source_domains)
source_domains.sort()
for val in source_domains:
    domains[val] = i
    i = i + 1

domains_bc = sc.broadcast(domains)


def source_index(source):
    return domains_bc.value[source]


udf_source_index = F.udf(source_index)

clean_data_sdf = clean_data_sdf.withColumn('source_index', udf_source_index('source_domain').cast('int'))

clean_data_sdf = clean_data_sdf[['source_domain', 'text_or_desc', 'clean_text', 'published_date', 'year', 'month',
                         'title', 'url', 'weeks' , 'tokens', 'masks', 'segments', 'ids', 'source_index']]

domain_lookup = pd.Series(domains)

log_time("Convert to pandas")

clean_data_pdf = clean_data_sdf.toPandas()

log_time("Begin store")


store = pd.HDFStore(OUTPUT_NAME)
store['clean_data'] = clean_data_sdf.toPandas()
store['domain_lookup'] = domain_lookup

store.info()
store.close()



log_time("Finished")
