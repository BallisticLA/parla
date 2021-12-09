

def set_docstring(docstr):
    def assign(fn):
        fn.__doc__ = docstr
        return fn
    return assign
