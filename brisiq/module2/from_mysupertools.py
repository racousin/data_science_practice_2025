from mysupertools.tool.operation_a_b import multiply

# Test with numbers
assert multiply(4, 5) == 20
assert multiply(2.5, 4) == 10.0
assert multiply(-3, 7) == -21

# Test with non-numbers
assert multiply("a", 5) == "error"
assert multiply(4, "b") == "error"
assert multiply("hello", "world") == "error"

print("All tests passed!")