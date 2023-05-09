import pickle
from language_model import LanguageModel
from error_model import ErrorModel
import re
import numpy as np
from collections import Counter, defaultdict
import functools
import util
from trie import Trie
import sys

def build_and_save_language_model(lm):
    lm.build_from_file("queries_all.txt")
    util.save_obj(lm, 'lm')

def build_and_save_error_model(em):
    em.build_from_file("queries_all.txt")
    util.save_obj(em, 'em')

def build_and_save_char_model(cbm):
    cbm.build_from_file("queries_all.txt")
    util.save_obj(cbm, 'cbm')

if __name__ == '__main__':
    #print("Start CharBigramModel building", file=sys.stderr)
    #cbm = CharBigramModel()
    #build_and_save_char_model(cbm)

    print("Start LanguageModel building", file=sys.stderr)
    lm = LanguageModel()
    build_and_save_language_model(lm)

    #print("Start ErrorModel building", file=sys.stderr)
    #em = ErrorModel()
    #build_and_save_error_model(em)


