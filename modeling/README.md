# Modeling

Modeling is run after data wrangling has produced data files with tokens for the original news stories.

## Create the model instance

Run this command to start up the modeling instance:

    gcloud compute instances create model1 \
      --zone=us-east1-c  \
      --machine-type=n1-highmem-8 \
      --image-family=tf2-latest-gpu \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=250 \
      --maintenance-policy=TERMINATE \
      --accelerator="type=nvidia-tesla-t4,count=1" \
      --metadata="install-nvidia-driver=True"

## Set up the model instance

Connect to the model instance.  For example, here is the command I use:

    gcloud compute ssh harold_hlneal_com@model1
    
Once on the instance, copy the bert_model.py file from the modeling/code directory of this project into the home 
directory on the model instance.   For example, first copy from local to a bucket by running this command locally:

    gsutil cp code/bert_model.py gs://topic-sentiment-1/code/
    
Then from the model instance, I run:

    gsutil cp gs://topic-sentiment-1/code/bert_model.py .

## Running the model

Several constants are set near the top of the bert_model.py script.  You may
wish to change these to point to the locations you want for input and output data.

To run the script, on the model instance, do this:

    nohup python bert_model.py 2>&1 &

This will save any on-screen output to a file named nohup.out.  Most on-screen output is also
logged.

## Shutting down the model instance

Once the modeling is complete, you can destroy the model instance.

    gcloud compute instances delete model1
    
You can return to the Airflow deploy section to deploy the model results.

        