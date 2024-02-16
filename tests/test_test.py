from pathlib import Path
from shutil import rmtree

from kpconv_torch.utils import tester


def test_get_test_save_path(fixture_path):
    assert tester.get_test_save_path(infered_file=None, trained_model=None) is None
    log_dir = Path(__file__).parent / "Log_test"
    infered_file = fixture_path / "example.ply"
    test_path = tester.get_test_save_path(infered_file=infered_file, trained_model=log_dir)
    assert test_path == fixture_path / "test" / "Log_test"
    assert test_path.exists()
    rmtree(test_path)
    test_path = tester.get_test_save_path(infered_file=None, trained_model=log_dir)
    assert test_path == Path(log_dir) / "test"
    assert test_path.exists()
    rmtree(test_path)
