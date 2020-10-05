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

