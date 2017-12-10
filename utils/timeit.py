from __future__ import print_function

from time import time
from functools import wraps
from contextlib import contextmanager

def timeit(func):
    @wraps(func)
    def runner(*arg, **kw):
        start_time = time()
        result = func(*arg, **kw)
        end_time = time()
        print('finish in {} seconds\n'.format(end_time - start_time))
        return result
    return runner

@contextmanager
def timeit_context(*args):
    start_time = time()
    yield
    end_time = time()
    if args:
        print(*args)
    print('finish in {} seconds\n'.format(end_time - start_time))

__all__ = ['timeit', 'timeit_context']
