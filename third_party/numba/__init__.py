# Minimal numba stub for environments without numba installed.
# Provides a no-op njit decorator compatible with usages in pandas_ta.

def njit(signature_or_function=None, **kwargs):
    if callable(signature_or_function):
        # Used as @njit without args
        return signature_or_function

    # Used as @njit(...)
    def _decorator(func):
        return func

    return _decorator
