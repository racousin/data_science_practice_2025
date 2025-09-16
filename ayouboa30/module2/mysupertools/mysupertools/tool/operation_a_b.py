def multiply(a, b):
    """
    Multiply two values if they are both numbers.

    Args:
        a: First value
        b: Second value

    Returns:
        Product of a and b if both are numbers, "error" otherwise
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        raise ValueError("Ce ne sont pas des nombres")
