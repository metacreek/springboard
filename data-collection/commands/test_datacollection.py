from unittest import mock
from datetime import date
import datacollection as dc


def test_start_date():
    with mock.patch('datacollection.date') as mock_date:
        mock_date.today.return_value = date(2020, 1, 10)

        assert dc.start_date(before=1) == '2020-01-09'

