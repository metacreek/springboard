from datetime import date, timedelta
import os

BASEDIR = '/home/ec2-user/news-please-repo'


def start_date(before=0):
    start = date.today() - timedelta(days=before)
    return str(start)


def create_config_file():
    with open(f'{BASEDIR}/config/config_base.cfg') as infile, open(f'{BASEDIR}/config/config.cfg', 'w') as outfile:
        for line in infile:
            if line.startswith('start_date'):
                line = f"start_date = '{start_date(before=8)} 00:00:00'\n"
            elif line.startswith('LOG_FILE'):
                line = f"LOG_FILE = '{BASEDIR}/log_{start_date()}.txt'\n"
            outfile.write(line)


def get_crawlers():
    f = os.popen("ps ax | grep single_crawler | grep -v grep | cut -d ' '  -f 2", 'r', 10240)
    values = f.read().split('\n')
    if '' in values:
        values.remove('')
    return values


def count_crawlers():
    p = get_crawlers()
    return len(p)


def kill_crawlers():
    crawl = get_crawlers()
    for c in crawl:
        command = f'kill -9 {c}'
        print(command)
        os.system(command)


def start_crawlers():
    os.system('news-please &')





