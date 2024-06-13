import threading
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def run_with_max_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract max_wait from the function's arguments
        max_wait = kwargs.pop("max_wait", None)

        if max_wait is None:
            # warn if max_wait is not provided.
            logger.warning(
                f"max_wait not provided for function {func.__name__}. Defaulting to 60 seconds."
            )
            max_wait = 60

        # This function will run the target function
        def target(result, *args, **kwargs):
            result[0] = func(*args, **kwargs)

        result = [None]
        thread = threading.Thread(target=target, args=(result, *args), kwargs=kwargs)

        thread.start()
        thread.join(max_wait)

        if thread.is_alive():
            return f"Function {func.__name__} did not finish in {max_wait} seconds and was terminated."

        return result[0]

    return wrapper
