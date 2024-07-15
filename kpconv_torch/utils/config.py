import yaml

from pathlib import Path


def save_config(train_save_path, config):
    with open(Path(train_save_path / "config.yml"), "w") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path):
    if file_path is None:
        file_path = "config.yml"
    else:
        file_path = Path(file_path)

    with open(file_path) as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    return config
