def is_jupyter_notebook():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False
