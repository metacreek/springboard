import datacollection as dc
import time

# Kill all existing crawler processes

while dc.count_crawlers() > 0:
    dc.kill_crawlers()
    time.sleep(5)