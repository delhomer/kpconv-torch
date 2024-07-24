import yaml

from pathlib import Path


def save_config(train_save_path, config):
    with open(Path(train_save_path / "config.yml"), "w") as file_object:
        yaml.dump(config, file_object)


def load_config(file_path):
    file_path = Path(file_path)

    with open(file_path) as file_object:
        config = yaml.load(file_object, Loader=yaml.SafeLoader)

    # Check if dataset exists
    dataset = config["dataset"]
    if dataset not in ("ModelNet40", "NPM3D", "S3DIS", "SemanticKitti", "Toronto3D"):
        raise ValueError(
            f"Error - unsupported dataset {dataset}, must be\
            among ModelNet40, NPM3D, S3DIS, SemanticKitti, Toronto3D"
        )
    return config
