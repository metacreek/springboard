import time
#import gsutilwrap
#import os

BUCKET = 'gs://topic-sentiment-1'  # bucket for data
LOCAL_DATA_DIR = 'data'  # relative name of local directory where input files will be copied
LOCAL_MODEL_DIR = 'model'  # relative name of local directory where final model is saved
LOCAL_CHECKPOINT_DIR = 'checkpoint'  # relative name of local directory where checkpoint files are saved
LOCAL_TEST_OUTPUT_DIR = 'output'  # relative name of local directory where supplementary output is saved
TOKENIZED_DATA_DIR = 'test_tokenized'  # name of bucket subdirectory where input files will be copied from
BUCKET_RESULTS_DIR = 'test-docker'  # name of bucket subdirectory where output files will be copied to

#os.mkdir('data')
#gsutilwrap.copy(pattern=f"{BUCKET}/{TOKENIZED_DATA_DIR}/*", target=f"{LOCAL_DATA_DIR}/", recursive=True)

#gsutilwrap.copy(pattern=f"{LOCAL_DATA_DIR}/*", target=f"{BUCKET}/{BUCKET_RESULTS_DIR}", recursive=True)

for i in range(10000):
    print(i)
    with open('output.txt', 'a') as f:
        f.write(f"{i}\n")
    time.sleep(0.5)
