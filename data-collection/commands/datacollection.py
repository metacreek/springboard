from datetime import date, timedelta
import os
import re

# This module contains utility functions for collection of news stories

# Set BASEDIR to the home directory for data collection
BASEDIR = '/news-please-repo'


def start_date(before=0):
    """
    Returns a date as a string that is 'before' days before the current date
    Used to specify the start_date in the config file for crawling

    :param before: integer, default 0, how many days before now should the start_date be
    """
    start = date.today() - timedelta(days=before)
    return str(start)


def create_config_file(before=10, user_agent='default', sitelist='sitelist'):
    """
    Creates config.cfg from the base config_base.cfg file according to parameters.
    Assumes the presence of BASEDIR/config/config_base.cfg

    :param before: integer, number of days before today to set as the crawl start_date
    :param user_agent: string, either 'google' for google crawler as user agent or 'default' for news-please user agent
    :param sitelist: string, specifies which sitelist file to use
    """
    with open(f'{BASEDIR}/config/config_base.cfg') as infile, open(f'{BASEDIR}/config/config.cfg', 'w') as outfile:
        for line in infile:
            if line.startswith('start_date'):
                line = f"start_date = '{start_date(before)} 00:00:00'\n"
            elif line.startswith('end_date'):
                line = f"end_date = '{start_date(-10)} 00:00:00'\n"
            elif line.startswith('LOG_FILE'):
                line = f"LOG_FILE = '{BASEDIR}/log_{start_date()}_{user_agent}_{sitelist}_{before}.txt'\n"
            elif line.startswith('USER_AGENT'):
                if user_agent == 'default':
                    line = f"USER_AGENT = 'news-please (+http://www.example.com/)'"
                elif user_agent == 'googlebot':
                    line = f"USER_AGENT = 'Googlebot'"
                else:
                    print("No USER_AGENT set!!!")
            elif line.startswith('url_input_file_name'):
                line = f"url_input_file_name = {sitelist}.hjson"
            outfile.write(line)


def get_crawlers():
    """
    Returns list of process ids for crawler processes
    """
    f = os.popen("ps ax | grep single_crawler | grep -v grep | cut -c 1-5", 'r', 10240)
    values = f.read().split('\n')
    if '' in values:
        values.remove('')
    return values


def count_crawlers():
    """
    Returns number of crawlers currently running
    """
    p = get_crawlers()
    return len(p)


def kill_crawlers():
    """
    Kills all crawler processes
    """
    crawl = get_crawlers()
    for c in crawl:
        command = f'kill -9 {c}'
        print(command)
        os.system(command)


def start_crawlers():
    """
    Starts the news-please crawling service in the background
    """
    os.system('news-please -c /news-please-repo/config/')


def combine():
    """
    Combines individual story files from news-please into a single file
    """
    with open(f'{BASEDIR}/config/config.cfg') as infile:
        for line in infile:
            if line.startswith('LOG_FILE'):
                match = re.split('(\d{4}\-\d{1,2}\-\d{1,2})_([A-Za-z]+)_(\w+)_(\d+).txt', line)
                if match:
                    output_file = f"{BASEDIR}/combined/{match[1]}_{match[2]}_{match[3]}_{match[4]}.json"
                    print(output_file)
                    cmd = f"find /home/ec2-user/news-please-repo/data/ -name '*.json' -print0 | xargs --null sed -e 's/./&/' -s > {output_file}"
                    os.system(cmd)
                else:
                    print("The output file name cannot be determined")
                    exit(-1)

