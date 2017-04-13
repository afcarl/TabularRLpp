import os


def merge(dir1, dir2):
    dirs1 = os.listdir(dir1)
    for i in dirs1:
        f1 = open(dir1 + i, 'r')
        f2 = open(dir2 + i, 'a+')
        contents = f1.read()
        f2.write('\n' + contents + '\n')
        f1.close()
        f2.close()