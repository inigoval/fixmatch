import yaml
from paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def update_config(config):
    """
    Adjust config values if using rgz data to remove obsolete values
    """
    if config["type"] == "fixmatch":
        if config["data"]["u"] != "rgz":
            del config["cut_threshold"]

        if config["mu"] == -1:
            config["mu"] = config["data"]["u_frac"]

        if config["data"]["u"] == "rgz":
            config["data"]["fri_R"] = -1

    if config["type"] == "baseline":
        config["data"]["fri_R"] = -1
        config["data"]["u"] = "all"
        config["mu"] = 1
        config["data"]["u_frac"] = 1
        config["train"]["p-strong"] = 0
        config["cutout"] = 0
        config["randpixel"] = 0

        del config["cut_threshold"]
        del config["lambda"]
        del config["tau"]


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""
    path = path_dict["root"] / "config.yml"
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    update_config(config)
    return config
