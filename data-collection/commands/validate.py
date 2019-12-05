import json

line_no = 0
with open('/home/ec2-user/news-please-repo/combined/2019-11-26a.json') as f:
    for line in f:
        line_no = line_no + 1
        try:
            json.loads(line)
            break
        except:
            print(f'Line {line_no} could not be read: ', line)



