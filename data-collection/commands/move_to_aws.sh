#!/bin/bash

gzip -v /home/ec2-user/news-please-repo/*.txt
aws s3 cp /home/ec2-user/news-please-repo/*.txt.gz s3://topic-sentiment-1/crawl-logs/
aws s3 cp /home/ec2-user/news-please-repo/combined/*.json s3://topic-sentiment-1/combined/