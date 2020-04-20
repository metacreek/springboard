import datacollection as dc
import time

# Kill any existing crawler process and begin a new data collection instance

while dc.count_crawlers() > 0:
    dc.kill_crawlers()
    time.sleep(5)
dc.create_config_file(before=400, user_agent='googlebot', sitelist='sitelist_googlebot_b')
dc.start_crawlers()

