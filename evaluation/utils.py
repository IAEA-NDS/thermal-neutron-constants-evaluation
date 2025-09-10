import time


def timeit(func):
    def wrapper(*args, **kwargs):
        s1 = time.time()
        ret = func(*args, **kwargs)
        s2 = time.time()
        print(f'time: {s2-s1}')
        return ret
    return wrapper
