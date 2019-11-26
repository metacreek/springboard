try:
    from . import topicsentiment as ts
except:
    import topicsentiment as ts
import time

while ts.count_crawlers() > 0:
    ts.kill_crawlers()
    time.sleep(5)