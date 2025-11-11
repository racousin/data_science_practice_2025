from numbers import Number
from typing import Any, Union


def multiply(a: Any, b: Any) -> Union[float, int, str]:
    """Multiply two numeric values.

    - Returns the arithmetic product when both inputs are numeric
      (``int``/``float`` and other ``numbers.Number`` subtypes).
    - Returns the string "error" otherwise (keeps existing behavior
      expected by tests).

    Args:
        a: First operand
        b: Second operand

    Returns:
        Product of ``a`` and ``b`` if both are numbers, otherwise "error".
    """
    if isinstance(a, Number) and isinstance(b, Number):
        return a * b
    return "error"
