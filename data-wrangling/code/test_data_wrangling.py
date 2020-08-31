import pytest
import os
#@mock.patch.dict(os.environ, {'GOOGLE_APPLICATION_CREDENTIALS': 'mocked'})
import data_wrangling
from pyspark import SparkContext
from pyspark.sql import SQLContext
from datetime import datetime

spark_context = SparkContext()
sql_context = SQLContext(spark_context)

SUBSTITUTION_BC = ''

def get_value(sdf, colname):
    return sdf.select(colname).collect()[0][0]


def test_drop_columns():
    input_sdf = sql_context.createDataFrame(
        [
            ('test1', 'test1', 'test1', 'test1', 'test1', 'test1')
        ],
        [
            'filename',
            'image_url',
            'localpath',
            'title_page',
            'title_rss',
            'url'
        ]
    )
    removed_columns = ['filename',
            'image_url',
            'localpath',
            'title_page',
            'title_rss']
    output_sdf = data_wrangling.drop_columns(input_sdf)
    for col in removed_columns:
        assert col not in output_sdf.columns
    assert 'url' in output_sdf.columns


def test_add_published():
    input_sdf = sql_context.createDataFrame(
        [
            (1, "2019-01-07 22:00:00")
        ],
        [
            'id',
            'date_publish'
        ]
    )
    output_sdf = data_wrangling.add_published(input_sdf)
    assert 'published' in output_sdf.columns
    type = [pairs[1] for pairs in output_sdf.dtypes if pairs[0] == 'published']
    assert type.pop() == "timestamp"
    value = get_value(output_sdf.where(output_sdf.id == 1), 'published')
    assert value == datetime(2019, 1, 7, 22, 0)


def test_handle_language():
    input_sdf = sql_context.createDataFrame(
        [
            (1, 'some text', 'description', 'en', 'cnn.com'),
            (2, None, 'description', 'en', 'time.com'),
            (3, 'more', 'stuff', None, 'time.com'),
            (4, 'more', 'stuff', None, 'bbc.com')
        ],
        [
            'id',
            'text',
            'description',
            'language',
            'source_domain'
        ]
    )
    output_sdf = data_wrangling.handle_language(input_sdf)
    for row in output_sdf.rdd.collect():
        assert row.text_or_desc == row.text or row.text_or_desc == row.description
    assert get_value(output_sdf.where(output_sdf.id == 3), 'language_guess') == 'en'
    assert output_sdf.where(output_sdf.id == 4).count() == 0

def test_drop_empty():
    input_sdf = sql_context.createDataFrame(
        [
            (1, None, None, 'text1', 'title1'),
            (2, "2019-01-07 22:00:00", datetime(2019, 1, 7, 22, 0), None, 'title2'),
            (3, "2019-01-07 22:00:00", datetime(2019, 1, 7, 22, 0), 'text3', None),
            (4, "2019-01-07 22:00:00", datetime(2019, 1, 7, 22, 0), 'text4', 'title4')
        ],
        [
            'id',
            'date_publish',
            'published',
            'text_or_desc',
            'title'
        ]
    )
    output_sdf = data_wrangling.drop_empty(input_sdf)
    assert output_sdf.count() == 1
    assert get_value(output_sdf, 'id') == 4


def test_remove_duplicates():
    input_sdf = sql_context.createDataFrame(
        [
            (1, "2019-01-07 22:00:00", "http://link1", "text1"),
            (2, "2019-01-07 22:00:00", "http://link1", "text2"),
            (3, "2019-01-07 22:00:00", "http://link2", "text3"),
            (4, "2019-01-10 22:00:00", "http://link3", "text3")
        ],
        [
            'id',
            'date_publish',
            'url',
            'text_or_desc',
        ]
    )
    output_sdf = data_wrangling.remove_duplicates(input_sdf)
    assert output_sdf.where(output_sdf.url == 'http://link1').count() == 1
    assert output_sdf.where(output_sdf.text_or_desc == 'text3').count() == 1


def test_year_filter():
    input_sdf = sql_context.createDataFrame(
        [
            (1, 2010),
            (2, 2011),
            (3, 2009),
        ],
        [
            'id',
            'year',
        ]
    )
    output_sdf = data_wrangling.year_filter(input_sdf)
    assert output_sdf.count() == 2

def test_level_data():
    input_sdf = sql_context.createDataFrame(
        [
            (1, 'a.com', datetime(2019, 1, 7, 21, 0)),
            (2, 'a.com', datetime(2019, 1, 7, 23, 0)),
            (3, 'a.com', datetime(2019, 1, 7, 22, 10)),
            (4, 'a.com', datetime(2019, 1, 7, 22, 20)),
            (5, 'a.com', datetime(2019, 1, 7, 22, 30)),
            (6, 'b.com', datetime(2019, 1, 7, 22, 0)),
        ],
        [
            'id',
            'source_domain',
            'published'
        ]
    )
    output_sdf = data_wrangling.level_data(input_sdf, 3)
    assert output_sdf.where(output_sdf.source_domain == 'b.com').count() == 0
    assert output_sdf.where(output_sdf.source_domain == 'a.com').count() == 3
    for row in output_sdf.rdd.collect():
        assert row.published > datetime(2019, 1, 7, 22, 0)


def test_setup_regex_cleanup():
    substitution = data_wrangling.setup_regex_cleanup()
    for key, value in substitution.items():
        assert key in value

def test_clean_text():
    text = "the long brown horse"
    reg = "long"
    arr = [text, reg]
    result = data_wrangling.clean_text(arr)

def test_add_regex():
    regex = data_wrangling.add_regex('msnbc.com')
    assert regex == '(MSNBC|msnbc.com|\n)'