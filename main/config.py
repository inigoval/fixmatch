import yaml
from paths import Path_Handler

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def update_config(config):
    """
    Adjust config values if using rgz data to remove obsolete values
    """
    # TODO only use cut_threshold as an arg for save directory when using rgz dataset - modify in train script. Then can safely delete here.
    if config["method"] == "fixmatch":
        if config["data"]["unlabelled"] != "rgz":
            del config["cut_threshold"]

        if config["mu"] == -1:
            config["mu"] = config["data"]["u_frac"]

        if config["data"]["unlabelled"] == "rgz":
            config["data"]["fri_R"] = -1

    if config["method"] == "baseline":
        config["data"]["fri_R"] = -1
        config["data"]["unlabelled"] = "all"
        config["mu"] = 1
        config["data"]["u_frac"] = 1
        config["train"]["p-strong"] = 0
        config["cutout"] = 0
        config["randpixel"] = 0

        del config["cut_threshold"]  # used to specify save directory
        del config["lambda"]
        del config["tau"]

        # TODO extend to update for GZ MNIST?


def load_config():
    """Helper function to load yaml config file, convert to python dictionary and return."""
    path = path_dict["root"] / "config.yml"
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)

    update_config(config)
    return config
