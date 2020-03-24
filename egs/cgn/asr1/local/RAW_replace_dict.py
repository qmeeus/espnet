#!/usr/bin/env python3
import sys
from pathlib import Path

"""
Usage: replace_dict.py input_file dict_filed

input_file: freq index
dict_file: word index
"""

def parse(text, types=None):
    lines = text.split("\n")
    values = filter(lambda l: len(l) == 2, map(str.split, lines))
    if types:
        values = map(
            lambda vals: tuple(typ(v) for typ, v in zip(types, vals)),
            values
        )
    return {key: value for value, key in values}

input_file, dict_file = map(Path, sys.argv[1:])
with open(input_file) as infile, open(dict_file) as dictfile:
    frequencies =  parse(infile.read(), (int, int))
    index2word = parse(dictfile.read(), (str, int))
    for index, freq in frequencies.items():
        print(f"{index2word[index]} {freq}")


