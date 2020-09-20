import datacollection as dc
import time
import os

# Kill any existing crawler process and begin a new data collection instance

while dc.count_crawlers() > 0:
    dc.kill_crawlers()
    time.sleep(5)
before = int(os.getenv('BEFORE', 400))
before = int(before)
user_agent = os.getenv('USER_AGENT', 'googlebot')
sitelist = os.getenv('SITELIST', 'sitelist_googlebot_b')
dc.create_config_file(before=before, user_agent=user_agent, sitelist=sitelist)
dc.start_crawlers()

