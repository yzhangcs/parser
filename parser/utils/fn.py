# -*- coding: utf-8 -*-

import unicodedata


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A']
               for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char)
               for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char)
               for char in token)


def isprojective(sequence):
    for i in range(1, len(sequence)):
        hi = sequence[i]
        for j in range(i + 1, hi):
            hj = sequence[j]
            if hi >= 0 and hj >= 0 and (hj - hi) * (hj - i) > 0:
                return False
    return True


def numericalize(sequence):
    return [int(i) for i in sequence]
