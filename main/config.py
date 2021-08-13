import yaml
from paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def update_config(config):
    """
    Adjust config values if using rgz data to remove obsolete values
    """
    if config["data"]["u"] != "rgz":
        del config["cut_threshold"]

    if config["mu"] == -1:
        config["mu"] = config["data"]["u_frac"]


def load_config():
    """
    Helper function to load config file
    """
    path = path_dict["root"] / "config.yml"
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    update_config(config)
    return config
