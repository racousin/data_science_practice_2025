# mysupertools/tests/test_multiplication.py

from mysupertools.tool.operation_a_b import multiply


def test_multiply_numbers():
    assert multiply(4, 5) == 20
    assert multiply(-1, 5) == -5


def test_multiply_errors():
    assert multiply("a", 5) == "error"
    assert multiply(None, 5) == "error"
