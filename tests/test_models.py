"""Model shape / parameter / receptive-field checks (needs PyTorch; skipped if absent)."""
import pytest

torch = pytest.importorskip("torch")

from phonrec.models.blocks import count_params, output_lengths  # noqa: E402
from phonrec.models.models import ModelA, ModelB  # noqa: E402

# Exact counts of THIS implementation (convs without bias + BatchNorm), by arithmetic:
EXPECTED_A = 134_569   # spec design target ~131K (+2.7 %)
EXPECTED_B = 2_318_889  # spec design target ~2.3M (+0.8 %)


def test_param_counts():
    a, b = count_params(ModelA()), count_params(ModelB())
    print(f"\nModel A params = {a:,} (spec ~131K)   Model B params = {b:,} (spec ~2.3M)")
    assert a == EXPECTED_A
    assert b == EXPECTED_B
    assert abs(a / 131_000 - 1) < 0.05 and abs(b / 2_300_000 - 1) < 0.05


@pytest.mark.parametrize("cls", [ModelA, ModelB])
@pytest.mark.parametrize("T", [1, 2, 99, 100, 301])
def test_output_shape(cls, T):
    m = cls().eval()
    x = torch.randn(3, 42, T)
    lp, ol = m(x, torch.tensor([T, T, max(1, T - 1)]))
    assert lp.shape == (3, (T + 1) // 2, 41)
    assert ol.tolist()[0] == (T + 1) // 2
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3, (T + 1) // 2), atol=1e-4)


def test_output_lengths_formula():
    L = torch.arange(1, 50)
    assert (output_lengths(L) == (L + 1) // 2).all()


def test_receptive_field_analytic():
    assert ModelA().receptive_field() == 53     # 530 ms at 10 ms hop
    assert ModelB().receptive_field() == 53


def test_receptive_field_by_gradient():
    """Measure which input frames influence one output frame of Model A."""
    m = ModelA().eval()
    T = 201
    x = torch.randn(1, 42, T, requires_grad=True)
    lp, _ = m(x)
    t_out = lp.shape[1] // 2
    lp[0, t_out].sum().backward()
    g = x.grad.abs().sum(1)[0]
    nz = torch.nonzero(g > 0).flatten()
    span = int(nz.max() - nz.min() + 1)
    print(f"\nModel A measured receptive field = {span} frames; "
          f"future frames used = {int(nz.max()) - 2 * t_out}")
    assert span == 53
