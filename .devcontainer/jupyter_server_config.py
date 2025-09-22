c = get_config()  # noqa: F821
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = "/home/jovyan"
c.ServerApp.preferred_url = "/lab"
c.ServerApp.token = "dev"   # use '' to disable auth (not recommended)
