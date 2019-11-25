try:
    from . import topicsentiment as ts
except:
    import topicsentiment as ts

if ts.count_crawlers() > 0:
    ts.kill_crawlers()
ts.create_config_file()
ts.start_crawlers()