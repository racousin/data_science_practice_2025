# mysupertools/tests/test_multiplication.py


def test_multiply_numbers():
    from mysupertools.tool.operation_a_b import multiply
    assert multiply(4, 5) == 20
    assert multiply(-1, 5) == -5


def test_multiply_errors():
    from mysupertools.tool.operation_a_b import multiply
    assert multiply("a", 5) == "error"
    assert multiply(None, 5) == "error"

