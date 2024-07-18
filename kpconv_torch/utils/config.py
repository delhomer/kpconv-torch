import yaml

from pathlib import Path


def save_config(train_save_path, config):
    with open(Path(train_save_path / "config.yml"), "w") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path, dataset):
    if file_path is None:
        file_path = "config.yml"
    else:
        file_path = Path(file_path)

    with open(file_path) as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    if config["dataset"] != dataset:
        t1 = config["model"]["dataset"]
        raise ValueError(f"Config dataset ({t1}) " f"does not match provided dataset ({dataset}).")

    return config
