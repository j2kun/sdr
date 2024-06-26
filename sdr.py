"""Algorithms for computing signed digit representations"""


def looping_naf(x: int) -> list[int]:
    """Compute the base-2 non-adjacent form (NAF) of x.

    Args:
        x: the number to be decomposed.

    Returns:
        A list of integers representing the non-adjacent form of x,
        in order from most significant to least significant.
    """
    naf = []
    while x > 0:
        if x % 2 == 1:
            naf.append(2 - x % 4)
            x -= naf[-1]
        else:
            naf.append(0)
        x //= 2
    return list(reversed(naf))


def looping_recompose(naf):
    x = 0
    place_value = 1
    for digit in reversed(naf):
        x += digit * place_value
        place_value *= 2
    return x


def prodinger_naf(x) -> tuple[int, int]:
    """A more efficient version of looping_naf.

    Args:
        x: the number to be decomposed.

    Returns:
        Two integers with bits set corresponding to the positive and negative
        bits of the non-adjacent form of x. The first integer tracks the
        negative bits, the second the positive bits.
    """
    xh = x >> 1
    x3 = x + xh
    c = xh ^ x3
    positive_bits = x3 & c
    negative_bits = xh & c
    return (negative_bits, positive_bits)


def prodinger_recompose(neg, pos):
    x = 0
    place_value = 1
    while neg > 0 or pos > 0:
        n = neg & 1
        p = pos & 1
        x += -n * place_value + p * place_value
        place_value *= 2
        neg >>= 1
        pos >>= 1
    return x
