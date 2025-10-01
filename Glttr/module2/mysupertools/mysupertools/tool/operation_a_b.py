import numbers


def multiply(a, b):
    """
    Multiply two values if they are both numbers.

    Args:
        a: First value
        b: Second value

    Returns:
        Product of a and b if both are numbers, "error" otherwise
    """
    if isinstance(a, numbers.Number) & isinstance(b , numbers.Number):
        return a*b
    return "error"
    