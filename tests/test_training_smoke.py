"""Full CPU smoke run of training/resume/evaluation (needs PyTorch, ~1-3 min)."""
import pytest

pytest.importorskip("torch")


@pytest.mark.slow
def test_smoke_pipeline():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("smoke", root / "scripts" / "smoke_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main("smoke_out_pytest")
