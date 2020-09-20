
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

## Production code

Data wrangling code is in the code subdirectory.  This include pytest code for unit tests of data wrangling routines.  
Data wrangling is performed as part of the springboard_capstone Airflow DAG.
See the airflow directory for information on running the code.