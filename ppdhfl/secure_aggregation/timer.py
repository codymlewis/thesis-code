"""
Wrapper class and methods for recording howling a method takes to perofrm a computation
"""

import time
from functools import wraps

import pandas as pd


class Timer:
    def __init__(self):
        self.times = []
        self.cur_time = {}
        self.jit_round = True

    def start(self, name):
        self.cur_time[name] = time.time()

    def end(self, name):
        self.cur_time[name] = time.time() - self.cur_time[name]

    def next(self):
        if self.jit_round:
            self.jit_round = False
        else:
            self.times.append(self.cur_time)
        self.cur_time = {}

    def results(self):
        return pd.DataFrame(self.times)


def timeit(f):
    "Time a single method call and return the output of the method"
    @wraps(f)
    def wrap(self, *args, **kw):
        self.timer.start(f.__name__)
        result = f(self, *args, **kw)
        self.timer.end(f.__name__)
        return result
    return wrap


def timeit_next(f):
    "Time a single method call and return the output of the method, add the time to a list"
    @wraps(f)
    def wrap(self, *args, **kw):
        self.timer.start(f.__name__)
        result = f(self, *args, **kw)
        self.timer.end(f.__name__)
        self.timer.next()
        return result
    return wrap
