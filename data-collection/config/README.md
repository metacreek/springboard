## Configuration files for collecting story data

### Config files

config_base.cfg is a full configuration file for the news-please.  This file will be the basis for a 
specific configuration file named config.cfg created by running the `start.py` command.

### Sitelist files

Sitelist files contain the sites that you want to crawl.  It is generally
best to have the number of sites per file to be less than the `number_of_parallel_crawlers` setting
in the config.cfg file, which is set to 24.  The number of parallel crawlers is best set based on 
your specific computing environment.

* sitelist.hjson: default sitelist.  not actually used in crawling
* sitelist_default.hjson: Sites to be crawled with the default news-please user agent
* sitelist_googlebot_a.hjson: first set of sites to be crawled with googlebot user agent
* sitelist_googlebot_b.hjson: second set of sites to be crawled with googlebot user agent

Note that some sites are duplicated between the sitelist_default.hjson and the two googlebot site lists.  For some sites,
I found it better to crawl using both user agents because this gave better sitewide coverage.
Any duplication in stories is handled during data wrangling.