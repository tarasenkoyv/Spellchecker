from spellchecker import Spellchecker
import pickle
from error_model import ErrorModel
from language_model import LanguageModel
from trie import Trie
import numpy as np
from fix_generators import join_generator, split_generator, word_generator
from fix_generators import keyboard_layout_generator
from fix_generators import preprocess_req
from classifiers import stat_clf
import sys
import copy
import time 
import util

def build_test():
    fix_requests = []
    none_fix_requests = []
    split_requests = []
    join_requests = []
    with open("queries_all.txt", 'r', encoding='utf-8') as file:
        for line in file:
            line = line.rstrip('\n')
            if '\t' in line:
                orig_req = line[:(line.index('\t'))]
                fix_req = line[(line.index('\t') + 1):]
                orig_tokens = [t for t in preprocess_req(orig_req) if not t.is_delim]
                fix_tokens = [t for t in preprocess_req(fix_req) if not t.is_delim]
                if len(fix_tokens) == len(orig_tokens):
                    fix_requests.append((orig_req, fix_req))
                elif len(fix_tokens) < len(orig_tokens):
                    join_requests.append((orig_req, fix_req))
                elif len(fix_tokens) > len(orig_tokens):
                    split_requests.append((orig_req, fix_req))
            else:
                none_fix_requests.append(line)

        util.save_obj(fix_requests, 'fix_requests')
        util.save_obj(none_fix_requests, 'none_fix_requests')
        util.save_obj(split_requests, 'split_requests')
        util.save_obj(join_requests, 'join_requests')

def test_fix(cnt_test, fix_requests):
    indices = list(range(len(fix_requests)))
    np.random.shuffle(indices)
    indices = indices[:cnt_test]
    cnt_errors = 0
    errors = []
    with open("error_fix_requests.txt", "a") as f:
        for i, idx in enumerate(indices):
            try:
                orig_req, fix_req = fix_requests[idx]
                result = spellchecker.correction(orig_req, max_candidates=5)
                if (result != fix_req):
                    cnt_errors += 1
                    f.write(orig_req + "\t" + fix_req + "\n")
            except:
                errors.append(idx)
                cnt_errors += 1
    acc = (cnt_test - cnt_errors) / cnt_test
    return errors, acc

if __name__ == '__main__':
    print("Start LanguageModel loading", file=sys.stderr)
    lm = util.load_obj('lm')

    print("Start ErrorModel loading", file=sys.stderr)
    em = util.load_obj('em')

    print("Start Trie building", file=sys.stderr)
    trie_spellcheck = Trie(em, lm)
    trie_spellcheck.build()
        
    print("Spellchecker init", file=sys.stderr)
    spellchecker = Spellchecker(lm, trie_spellcheck, stat_clf)

    print("Spellchecker start", file=sys.stderr)

    if False:
        while True:
            query = input()
            result = spellchecker.correction(query, max_candidates=10, iterations=2)        
            print(result)

    fix_requests = util.load_obj('fix_requests')
    split_requests = util.load_obj('split_requests')
    join_requests = util.load_obj('join_requests')

    errors = []
    acc_l = []
    for _ in range(100):
        iter_errors, acc = test_fix(1000, fix_requests)
        errors.extend(iter_errors)
        acc_l.append(acc)

    cnt_errors = 0
    start = time.time()
    for i, orig_req in enumerate(none_fix_requests[INDEX:INDEX + N]):
        result = spellchecker.correction(orig_req, max_candidates=6)
        if (result != orig_req):
            cnt_errors += 1
    print(str(N - cnt_errors) + "/" + str(N))
    end = time.time()
    print(str((end - start) / 60))

