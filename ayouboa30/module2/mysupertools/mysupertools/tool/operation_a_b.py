def multiply(a, b):
    """
    Multiply two values if they are both numbers.

    Args:
        a: First value
        b: Second value

    Returns:
        Product of a and b if both are numbers, "error" otherwise
    """
    if (type(a) == float or type(a)==int) and (type(a) == float or type(b) == int):
        return a*b
    else:
        raise raise ValueError("Ce ne sont pas des nombres")