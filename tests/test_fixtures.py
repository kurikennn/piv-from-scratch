"""The WebGPU parity fixtures must stay in sync with the Python processor."""

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from pivfs.piv import PIVConfig, run_piv

FIXTURES = Path(__file__).resolve().parents[1] / "web" / "fixtures" / "parity.json"


def _cases():
    return json.loads(FIXTURES.read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_parity_fixture_matches_python(case):
    shape = (case["height"], case["width"])
    img_a, img_b = (
        np.frombuffer(base64.b64decode(case[k]), np.uint8).reshape(shape).astype(float)
        for k in ("img_a", "img_b")
    )
    cfg = PIVConfig.single_pass(
        case["window"],
        estimator=case["estimator"],
        validate=False,
        search_radius=case["search_radius"],
        overlap=case["overlap"],
    )
    vf = run_piv(img_a, img_b, cfg)
    np.testing.assert_allclose(vf.u.ravel(), case["u"], atol=1e-8)
    np.testing.assert_allclose(vf.v.ravel(), case["v"], atol=1e-8)
    np.testing.assert_array_equal(vf.x.ravel(), case["x"])
