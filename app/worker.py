import os

names = open("names.txt").read().splitlines()

for name in names:
    os.system("python3 crawler.py -p '{}'".format(name))
