# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""

import operator
import warnings


def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _last_left_t_init(sigma):
    return {c: 0 for c in sigma}


def _edit_dist_step(lev, i, j, s1, s2, last_left, last_right, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and last_left > 0 and last_right > 0:
        d = lev[last_left - 1][last_right - 1] + i - last_left + j - last_right - 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype: int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # retrieve alphabet
    sigma = set()
    sigma.update(s1)
    sigma.update(s2)

    # set up table to remember positions of last seen occurrence in s1
    last_left_t = _last_left_t_init(sigma)

    # iterate over the array
    for i in range(len1):
        last_right = 0
        for j in range(len2):
            last_left = last_left_t[s2[j]]
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                last_left,
                last_right,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )
            if s1[i] == s2[j]:
                last_right = j + 1
            last_left_t[s1[i]] = i + 1
    return lev[len1][len2]

def _edit_dist_backtrace(lev):
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment = [(i, j)]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j))
    return list(reversed(alignment))


def edit_distance_align(s1, s2, substitution_cost=1):
    """
    Calculate the minimum Levenshtein edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost.
    For example, mapping "rain" to "shine" would involve 2
    substitutions, 2 matches and an insertion resulting in
    the following mapping:
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)]
    NB: (0, 0) is the start state without any letters associated
    See more: https://web.stanford.edu/class/cs124/lec/med.pdf

    In case of multiple valid minimum-distance alignments, the
    backtrace has the following operation precedence:

    1. Skip s1 character
    2. Skip s2 character
    3. Substitute s1 and s2 characters

    The backtrace is carried out in reverse string order.

    This function does not support transposition.

    :param s1, s2: The strings to be aligned
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :rtype: List[Tuple(int, int)]
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                0,
                0,
                substitution_cost=substitution_cost,
                transpositions=False,
            )

    # backtrace to find alignment
    alignment = _edit_dist_backtrace(lev)
    return alignment

stop_words_en = \
    ['i',
    'me',
    'my',
    'myself',
    'we',
    'our',
    'ours',
    'ourselves',
    'you',
    "you're",
    "you've",
    "you'll",
    "you'd",
    'your',
    'yours',
    'yourself',
    'yourselves',
    'he',
    'him',
    'his',
    'himself',
    'she',
    "she's",
    'her',
    'hers',
    'herself',
    'it',
    "it's",
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    "that'll",
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'but',
    'if',
    'or',
    'because',
    'as',
    'until',
    'while',
    'of',
    'at',
    'by',
    'for',
    'with',
    'about',
    'against',
    'between',
    'into',
    'through',
    'during',
    'before',
    'after',
    'above',
    'below',
    'to',
    'from',
    'up',
    'down',
    'in',
    'out',
    'on',
    'off',
    'over',
    'under',
    'again',
    'further',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'all',
    'any',
    'both',
    'each',
    'few',
    'more',
    'most',
    'other',
    'some',
    'such',
    'no',
    'nor',
    'not',
    'only',
    'own',
    'same',
    'so',
    'than',
    'too',
    'very',
    's',
    't',
    'can',
    'will',
    'just',
    'don',
    "don't",
    'should',
    "should've",
    'now',
    'd',
    'll',
    'm',
    'o',
    're',
    've',
    'y',
    'ain',
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    'ma',
    'mightn',
    "mightn't",
    'mustn',
    "mustn't",
    'needn',
    "needn't",
    'shan',
    "shan't",
    'shouldn',
    "shouldn't",
    'wasn',
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't"]
stop_words_ru = \
    ['и',
     'в',
     'во',
     'не',
     'что',
     'он',
     'на',
     'я',
     'с',
     'со',
     'как',
     'а',
     'то',
     'все',
     'она',
     'так',
     'его',
     'но',
     'да',
     'ты',
     'к',
     'у',
     'же',
     'вы',
     'за',
     'бы',
     'по',
     'только',
     'ее',
     'мне',
     'было',
     'вот',
     'от',
     'меня',
     'еще',
     'нет',
     'о',
     'из',
     'ему',
     'теперь',
     'когда',
     'даже',
     'ну',
     'вдруг',
     'ли',
     'если',
     'уже',
     'или',
     'ни',
     'быть',
     'был',
     'него',
     'до',
     'вас',
     'нибудь',
     'опять',
     'уж',
     'вам',
     'ведь',
     'там',
     'потом',
     'себя',
     'ничего',
     'ей',
     'может',
     'они',
     'тут',
     'где',
     'есть',
     'надо',
     'ней',
     'для',
     'мы',
     'тебя',
     'их',
     'чем',
     'была',
     'сам',
     'чтоб',
     'без',
     'будто',
     'чего',
     'раз',
     'тоже',
     'себе',
     'под',
     'будет',
     'ж',
     'тогда',
     'кто',
     'этот',
     'того',
     'потому',
     'этого',
     'какой',
     'совсем',
     'ним',
     'здесь',
     'этом',
     'один',
     'почти',
     'мой',
     'тем',
     'чтобы',
     'нее',
     'сейчас',
     'были',
     'куда',
     'зачем',
     'всех',
     'никогда',
     'можно',
     'при',
     'наконец',
     'два',
     'об',
     'другой',
     'хоть',
     'после',
     'над',
     'больше',
     'тот',
     'через',
     'эти',
     'нас',
     'про',
     'всего',
     'них',
     'какая',
     'много',
     'разве',
     'три',
     'эту',
     'моя',
     'впрочем',
     'хорошо',
     'свою',
     'этой',
     'перед',
     'иногда',
     'лучше',
     'чуть',
     'том',
     'нельзя',
     'такой',
     'им',
     'более',
     'всегда',
     'конечно',
     'всю',
     'между']