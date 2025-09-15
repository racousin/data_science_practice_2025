def multiply(a, b):
    return a * b if all(isinstance(x, (int, float)) for x in (a, b)) else "error"
