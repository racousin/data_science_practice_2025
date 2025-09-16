def multiply(a, b):
    """
    Multiply two values if they are both numbers.

    Args:
        a: First value
        b: Second value

    Returns:
        Product of a and b if both are numbers, "error" otherwise
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return "error"
    else:
        return (a*b)
    

