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
from fix_generators import preprocess_req
import nltk_util
from collections import defaultdict

def build_and_save_language_model(lm):
    lm.build_from_file("queries_all.txt")
    util.save_obj(lm, 'lm')

def build_and_save_error_model(em):
    em.build_from_file("queries_all.txt")
    util.save_obj(em, 'em')
    
if __name__ == '__main__':
    try:
        #print("Start ErrorModel building", file=sys.stderr)
        #em = ErrorModel()
        #build_and_save_error_model(em)

        print("Start LanguageModel building", file=sys.stderr)
        lm = LanguageModel()
        build_and_save_language_model(lm)

    except RuntimeError as err:
        print(err, file=sys.stderr)
    except:
        print("Unknown error", file=sys.stderr)

