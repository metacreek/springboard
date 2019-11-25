This section of the repository contains code and instructions used to collect the JSON article files from a set of websites.

## EC2 Configuration

Create an instance based on the **Anaconda3 2019.10 on Amazon Linux 20191018.1855** AMI.

Once the instance is created, get the public IP address and set an environment variables named EC_CRAWLER

Create rule(s) as necessary so that you have inbound ssh access from your local machine (port 22)

The secret file is springboard.pem and is sorted in ~/.aws

To connect to the crawler instance:

    ssh -i ~/.aws/springboard.pem ec2-user@$EC_CRAWLER

## Initial setup

Connect to the crawler instance and execute the following commands:

    mkdir news-please-repo
    pip install news-please

You must also run the `aws configure` command and set your secrets and region.

## Sync code

From your local **data-collection** directory, run:

    scp -r -i ~/.aws/springboard.pem config commands ec2-user@$EC_CRAWLER:/home/ec2-user/news-please-repo
    
## Run crawler

Currently, the crawler is run manually:

    cd ~/news-please-repo/commands
    python start.py

## Copy data to S3

In the following, my data is stored in the **topic-sentiment-1** S3 bucket.

From the home directory on the crawler run:

    time aws s3 sync --only-show-errors --exclude "*.html" --exclude ".resume_jobdir/*" /home/ec2-user/news-please-repo/ s3://topic-sentiment-1


