try:
    from . import topicsentiment as ts
except:
    import topicsentiment as ts

ts.combine()


# DATE = '2019-11-26'
# DATA_PATH = '/home/ec2-user/news-please-repo/data/'
#
# directories = os.listdir(DATA_PATH)
# count = 0
# for path in directories:
#     append = ' >> ' if count else ' > '
#     cmd = f'find {DATA_PATH}{path}/ -name "*.json" -print0 | xargs --null cat {append} {DATA_PATH}{DATE}.json'
#     print(cmd)
#     count = count + 1
#     os.system(cmd)





