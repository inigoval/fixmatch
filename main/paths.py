from pathlib import Path


class Path_Handler:
    def __init__(self):
        self.dict = {}

    def fill_dict(self):
        root = Path(__file__).resolve().parent.parent
        path_dict = {}
        path_dict["root"] = root

        path_dict["data"] = root / "data"
        path_dict["files"] = root / "files"

        path_dict["rgz"] = root / "data" / "rgz"
        path_dict["mb"] = root / "data" / "mb"

        self.dict = path_dict

    def create_paths(self):
        for path in self.dict.values():
            if not Path.exists(path):
                Path.mkdir(path)

    def _dict(self):
        self.fill_dict()
        self.create_paths()
        return self.dict
