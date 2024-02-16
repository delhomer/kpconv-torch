from pathlib import Path
from shutil import rmtree

from kpconv_torch.utils import trainer


def test_get_train_save_path():
    assert trainer.get_train_save_path(output_dir=None, trained_model=None) is None
    log_dir = Path(__file__).parent / "Log_test"
    train_path = trainer.get_train_save_path(output_dir=None, trained_model=log_dir)
    assert log_dir == train_path
    assert Path(train_path).exists()
    rmtree(log_dir)
    output_dir = Path(__file__).parent / "outputdir"
    train_path = trainer.get_train_save_path(output_dir=output_dir, trained_model=None)
    assert Path(train_path).exists()
    log_dirs = list(Path(output_dir).iterdir())
    assert len(log_dirs) == 1
    rmtree(output_dir)
