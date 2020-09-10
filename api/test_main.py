import pytest
import main
import os

MAX_SEQ_LEN = 256


def test_lookup():
    lookup = main.lookup()
    assert len(lookup) > 0
    assert type(lookup) == dict
    assert type(lookup[0]) == str


def test_sites():
    sites = main.sites()
    assert len(sites) > 0
    assert type(sites) == list
    assert type(sites[0]) == str


def test_get_stopwords():
    sw = main.get_stopwords()
    assert len(sw) > 0
    assert type(sw) == str
    assert "myself" in sw


def test_get_masks():
    masks = main.get_masks("The chalkboard is in the other room", MAX_SEQ_LEN)
    assert masks == 35 * [1] + 221 * [0]


def test_get_segments():
    segments = main.get_segments("The chalkboard is in the other room", MAX_SEQ_LEN)
    assert segments == 256 * [0]


def test_create_single_input():
    text = "For me myself, I would not look at it in that way."
    input = main.create_single_input(text, MAX_SEQ_LEN)
    assert input == (
        [101, 1010, 2298, 2126, 1012, 102] + 250 * [0],
        6 * [1] + 250 * [0],
        256 * [0]
    )


def test_get_next_highest():
    predictions = [[0.3, 0.2, 0.1]]
    next_highest = main.get_next_highest(predictions)
    assert next_highest == (0.3, 0, [0])


def test_lookup_path():
    val = main.lookup_path()
    assert val == 'prod1'
    os.environ["DOMAIN_LOOKUP_PATH"] = 'bbb'
    val = main.lookup_path()
    assert val == 'bbb'

