''' author: sam tenka
    credits:
    date: 2016-11-21
    descr: Helper decorators
    usage:
        from utils.algo import memoize
'''

from __future__ import print_function

def memoize(func):
    ''' Accelerate repeated calls to func by storing
        results of previous arguments. 

        Currently no support for keyword arguments.
    '''
    memo = {}
    def mfunc(*args):
        key = tuple(args)
        if key not in memo:
            memo[key] = func(*args)
        return memo[key] 
    return mfunc

def fail_gracefully(func):
    ''' Attempt func, but suppress exceptions. 
    '''
    def gfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs) 
        except Exception:
            pass
    return gfunc
