This contains a variety of ways to create dataproc clusters.
Different calls are used depending on the cluster needed.

# General startup without custom image

## Feature extraction

### Using standard machines
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --initialization-actions gs://topic-sentiment-1/pip-install.sh \
  --metadata 'PIP_PACKAGES=spark-nlp==2.4.1 contractions'

### alternate for bert approach  
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --num-workers 2 \
  --master-machine-type n1-highmem-4 \
  --worker-machine-type n1-highmem-4 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=tensorflow==2.0.0 pyarrow==0.15.1 sentencepiece==0.1.85 gcsfs nltk tensorflow-hub tables bert-for-tf2 absl-py google-cloud-storage'  

### testing alternate for bert approach  
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --num-workers 2 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=tensorflow==2.0.0 pyarrow==0.15.1 sentencepiece==0.1.85 gcsfs nltk tensorflow-hub tables bert-for-tf2 absl-py google-cloud-storage google-cloud-logging'  


### Setup allowing Jupyter usage
gcloud beta dataproc clusters create classic1 \
  --image-version 1.4.22-debian9 \
  --optional-components=ANACONDA,JUPYTER \
  --enable-component-gateway \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=contractions nltk' \
  --bucket topic-sentiment-1

### Setup for five workers 
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --num-workers 5 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=spark-nlp==2.4.4 contractions'



### Setup for limited disk quota
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --num-workers 5 \
  --master-boot-disk-size 600 --worker-boot-disk-size 600 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=spark-nlp==2.4.4 contractions'

### Setup for SSDs
gcloud dataproc clusters create features1 \
  --image-version 1.4.22-debian9 \
  --num-workers 3 \
  --worker-boot-disk-type=pd-ssd \
  --worker-boot-disk-size 100 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=spark-nlp==2.4.1 contractions'
  
  
### Setup AI instance
gcloud compute instances create model1 \
        --zone=us-east1-c \
        --image-family=tf-latest-cu92 \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=n1-standard-8 \
        --metadata="install-nvidia-driver=True"  


gcloud compute instances create model1 \
        --zone=us-east1-c \
        --image-family=tf-latest-cu92 \
        --image-project=topic-sentiment-269614 \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=n1-standard-8 \
        --metadata="install-nvidia-driver=True" 
        
gcloud compute instances create model1 \
        --zone=us-east1-c \
        --image-family=tf-latest-cu92 \
        --image-project=deeplearning-platform-release 
 
         

gcloud compute --project=topic-sentiment-269614 instances get-serial-port-output model1 --zone=us-east1-c

## Modeling clusters for Keras and Tensorflow

### Setup for single-node cluster with a powerful machine
gcloud dataproc clusters create model1 \
  --master-machine-type n1-highmem-32 \
  --image-version 1.4.22-debian9   \
  --single-node  \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh   \
  --metadata 'PIP_PACKAGES=keras==2.3.1 tensorflow==2.1.0'


### Setup for 5 workers
gcloud dataproc clusters create model1 \
  --image-version 1.4.22-debian9   \
  --num-workers 5  \
  --master-boot-disk-size 600 \
  --worker-boot-disk-size 600   \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh   \
  --metadata 'PIP_PACKAGES=keras==2.3.1 tensorflow==2.1.0'

# For building a custom image

I am documenting this approach even though I abandoned it.  The issue
that I had was that pre-downloading of the pre-trained models did not
help with the processing.

From local machine:

gcloud config set project topic-sentiment-269614

gcloud dataproc clusters create image-builder --single-node

From the compute instance

gcloud auth login

Upload custom image script and code from https://github.com/GoogleCloudDataproc/custom-images to a bucket created for image creation
This assumes you have these ingredients in the env-setup directory.

gsutil cp -r gs://custom-nlp-image/env-setup/* .

cd ~/custom-images

python generate_custom_image.py \
    --image-name  custom-nlp-image-v3 \
    --dataproc-version 1.4.22-debian9 \
    --customization-script ../setup.sh \
    --zone us-east1-d \
    --gcs-bucket gs://custom-nlp-image \
    --no-smoke-test

This takes quite a while.

Back on the local machine.

gcloud dataproc clusters create topic-sentiment-cluster-4 \
  --image=https://www.googleapis.com/compute/beta/projects/topic-sentiment-269614/global/images/custom-nlp-image-v3 \
  --initialization-actions gs://goog-dataproc-initialization-actions-us-east1/python/pip-install.sh \
  --metadata 'PIP_PACKAGES=spark-nlp==2.4.1'
