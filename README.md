# Springboard Capstone Project for Machine Learning Engineering

This repository documents my Springboard Machine Learning Engineer capstone project.  This 
project involved the full lifecycle of data collection, analysis, wrangling, and modeling. 
The project adapts a Bert-based model to be used for classification of text from 
dozens of publicans.  It was built using both AWS and Google Cloud.

For an description of the capstone project, click [here](https://github.com/metacreek/springboard/tree/master/CAPSTONE.md). 

* [airflow](https://github.com/metacreek/springboard/tree/master/airflow) contains code to use with Google Cloud Composer,
which is a hosted version of Apache Airflow.  This code manages the running of data wrangling, and
model and user interface deployment.

* [api](https://github.com/metacreek/springboard/tree/master/api) contains a Flask application
that presents a user interface that allows use of model for this project.  This application
is hosted through Google Cloud Functions and is deployed by Airflow.

* [data-collection](https://github.com/metacreek/springboard/tree/master/data-collection) contains code used to crawl 
dozens of news and opinion websites.

* [data-wrangling](https://github.com/metacreek/springboard/tree/master/data-wrangling) contains Jupyter notebooks 
outlining evaluation of the collected data and analysis use to wrangle the data into a usable format.  It also
contains a Python program used to clean, wrangle and prepare data for use in modeling. This program is run on PySpark 
via Google Dataproc.  The running of this program is managed by Airflow

* [modeling](https://github.com/metacreek/springboard/tree/master/modeling) contains code and notebooks used to fine tune a BERT model to classify news stories.

Other directories:

* [images](https://github.com/metacreek/springboard/tree/master/images) contains images used in this documentation.

* [mini-projects](https://github.com/metacreek/springboard/tree/master/mini-projects) contains Jupyter notebooks submitted as part of classwork for the Springboard Machine Learning Engineer Bootcamp.  This code is not directly related to the capstone project.

* [static](https://github.com/metacreek/springboard/tree/master/mini-projects) contains CSS and images needed by the
Flask-based user interface.  These must deployed to a Google Storage Bucket once.