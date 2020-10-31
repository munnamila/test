import os
import time


def mager_path(_list):

    new_path = '/'

    for i in _list:

        new_path = os.path.join(new_path, i)

    return new_path


def time_str():

    out = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    return out


if __name__ == '__main__':

    print(time_str())


