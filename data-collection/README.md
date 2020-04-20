This section of the repository contains code used to collect the JSON article files from a set of websites.

The crawling is done using the [news-please](https://github.com/fhamborg/news-please) package.

## EC2 Configuration

Create an instance based on the **Anaconda3 2019.10 on Amazon Linux 20191018.1855** AMI.

Once the instance is created, get the public IP address and set an environment variables named EC_CRAWLER

Create rule(s) as necessary so that you have inbound ssh access from your local machine (port 22)

The secret file is springboard.pem and is sorted in ~/.aws

Set EC_CRAWLER to the IP address of your instance

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

From the EC2 instance, in the commands sub-directory:

    ./move_to_aws.sh

