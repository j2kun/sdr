import pytest

from sdr import looping_naf
from sdr import looping_recompose
from sdr import prodinger_naf
from sdr import prodinger_recompose

@pytest.mark.parametrize("x", list(range(128)))
def test_looping_naf(x):
    assert looping_recompose(looping_naf(x)) == x

@pytest.mark.parametrize("x", list(range(128)))
def test_prodinger_naf(x):
    assert prodinger_recompose(*prodinger_naf(x)) == x
