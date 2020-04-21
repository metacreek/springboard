
# Data analysis and wrangling

This folder contains Jupyter Notebooks that were used while analyzing the data collected
and while figuring out how to wrangle data.

All data analysis was performed using [Apache Spark](https://spark.apache.org/) running on the 
[AWS EMR](https://aws.amazon.com/emr/) product.  The 
script [emr_bootstrap.sh](https://github.com/metacreek/springboard/blob/master/data-wrangling/emr_bootstrap.sh) was used as part
of the startup process for AWS EMR.

* [Initial exploratory data analysis](https://github.com/metacreek/springboard/blob/master/data-wrangling/EDA-initial.ipynb): 
The initial analysis was performed after the first run of data collection to get an idea of
the breadth of data across different sites.  I also looked at the languages in the results and
decided to focus on English language stories.  I looked at missing data and discovered duplication
that would need to be cleaned up.

* [User agent analysis](https://github.com/metacreek/springboard/blob/master/data-wrangling/EDA-user-agent.ipynb): This 
analysis was used to determine which crawler user agent should be used for each site.  In 
some cases, this led me to determine I should crawl with both user agents because
some sites returned a significant set of stories when either was used, and in
many cases these were not duplicates.  It seems that different results were returned depended on which
user agent was used.

* [Text duplicates analysis](https://github.com/metacreek/springboard/blob/master/data-wrangling/text-duplicates-analysis.ipynb): 
This analysis was used to debug a problem with duplicate text results.  From this,
changes were made to the data wrangling process.

* [Data wrangling](https://github.com/metacreek/springboard/blob/master/data-wrangling/Data-Wrangling.ipynb): This
notebook describes the pre-processing needed before moving data onto the modeling step. In this
notebook, you will note that I check the count of documents after nearly every step so that I could see
where data was being removed.  In a production system, I would not do this because these steps must
be immediately evaluated, which slows down the processing significantly.  Removing
the counts would mean that all the steps could be performed as part of one Spark operation.

## AWS EMR configuration

The following command shows the configuration used for AWS EMR:

    aws emr create-cluster --auto-scaling-role EMR_AutoScaling_DefaultRole --applications Name=Hadoop Name=Hive Name=Spark Name=Livy --bootstrap-actions '[{"Path":"s3://topic-sentiment-1/emr-scripts/emr_bootstrap.sh","Name":"Custom action"}]' --ebs-root-volume-size 10 --ec2-attributes '{"KeyName":"springboard","AdditionalSlaveSecurityGroups":["sg-098ceaa0b4e3e8753"],"InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-91129ac8","EmrManagedSlaveSecurityGroup":"sg-0282be9f7f446a030","EmrManagedMasterSecurityGroup":"sg-03c982f39c4d97925","AdditionalMasterSecurityGroups":["sg-e78bf783"]}' --service-role EMR_DefaultRole --enable-debugging --release-label emr-5.28.0 --log-uri 's3n://topic-sentiment-1/emr-logs/' --name 'topic-sentiment-v6' --instance-groups '[{"InstanceCount":2,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"CORE","InstanceType":"m5.xlarge","Name":"Core - 2"},{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":32,"VolumeType":"gp2"},"VolumesPerInstance":2}]},"InstanceGroupType":"MASTER","InstanceType":"m5.xlarge","Name":"Master - 1"}]' --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1