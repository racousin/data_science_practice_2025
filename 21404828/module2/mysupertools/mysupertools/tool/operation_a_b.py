
from typing import Union

Number = Union[int, float]

def multiply(a: Number, b: Number):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "error"
    
    return a * b