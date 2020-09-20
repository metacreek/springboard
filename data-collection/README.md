# Data Collection

Data collection is performed using the provided Docker container.  To build the container:

    docker build .

Note the image_id that is reported as built.  The crawling process takes three arguments, which are passed as environment
variables to the docker container.  These are:

* BEFORE:  limit the crawl to articles published after this many days before today.  See note below. 
* USER_AGENT: 'default' or 'googlebot'.  Identifies the user agent used for crawling.
* SITELIST: 'sitelist_default', 'sitelist_googlebot_a', or 'sitelist_googlebot_b'.  The sites to be crawled, as described in the configuration files built into the container.

Once the container is built, you can run it using:

    docker run [image_id] -e "BEFORE=20" -e "USER_AGENT=default" -e "SITELIST=sitelist_default"

Note that some site appear in multiple sitelist files because my analysis found those sites would report different pages 
depending on the user agent that was used.  
 
NOTE: the BEFORE argument specifies a start_date configuration parameter.
Unfortunately, the crawler treats this as a suggestion, so I generally found it necessary to kill the crawler after it had
crawled a sufficient number of pages.  To do this, connect to the docker container:

    docker exec -it [container_id] /bin/bash
    
Then inside the container run:

    python commands/kill_all.py
    
Finally, copy the data to the AWS buckets by running:

    ./commands/copy_to_aws.sh
    
The crawling is done using the [news-please](https://github.com/fhamborg/news-please) package.
