"""Tests for the Python 3.10 compatibility launcher without launching TH08."""
import hashlib
from io import BytesIO

from tools.rl.v400_retail_py310_launcher import _file_digest, main


def test_hashlib_shim_streams_exact_hash():
    data = b"th08" * 500
    assert _file_digest(BytesIO(data), "sha256", _bufsize=7).digest() == hashlib.sha256(data).digest()


def test_shim_self_test(capsys):
    assert main(["--shim-self-test"]) == 0
    assert "verified" in capsys.readouterr().out
