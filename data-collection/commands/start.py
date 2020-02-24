try:
    from . import topicsentiment as ts
except:
    import topicsentiment as ts
import time

while ts.count_crawlers() > 0:
    ts.kill_crawlers()
    time.sleep(5)
ts.create_config_file(before=400, user_agent='googlebot', sitelist='sitelist_googlebot_b')
ts.start_crawlers()

