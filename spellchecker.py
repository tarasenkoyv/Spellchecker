import pickle
from collections import defaultdict
from error_model import ErrorModel
from language_model import LanguageModel
from trie import Trie, Candidate
import numpy as np
from fix_generators import join_generator, split_generator, word_generator, join_generator_simple
from fix_generators import def_is_estimated_token, def_is_spec_join_token
from fix_generators import preprocess_req, keyboard_layout_generator, split_generator_complex
from classifiers import stat_clf
import sys
import copy 
import util

class Spellchecker:
    def __init__(self, language_model, trie, clf):
         self.language_model = language_model
         self.trie = trie
         self.clf = clf
         pass

    def safe_correction(self, orig_request, iterations=1, max_candidates=5):
        try:
            return self.correction(orig_request, iterations, max_candidates)
        except RuntimeError as err:
            print(err, file=sys.stderr)
        except:
            print("Error " + orig_request, file=sys.stderr)
        return orig_request

    def correction(self, orig_request, iterations=1, max_candidates=5):
        requests = set([orig_request])
        old_requests =  {}
        accumulated_errors = defaultdict(float)
        for i in range(iterations):
            new_requests = set()
            for req in requests:
                accumulated_error = accumulated_errors[req]
                tokens = preprocess_req(req)
                words = [t.token for t in tokens if t.need_correct]
                if len(words) == 0:
                    if req not in old_requests:
                        old_requests[req] = 0
                        new_requests.add(req)
                    continue

                fix_req_spec_join = def_is_spec_join_token(req)
                if fix_req_spec_join:
                    return fix_req_spec_join

                res = word_generator(tokens, self.language_model, self.trie, max_candidates)
                for fix_req_w, fix_list in res:
                    if fix_req_w not in old_requests:
                        req_error = self.clf(fix_list, self.language_model)
                        sum_error = sum([c.error_weight for c in fix_list])
                        accumulated_errors[fix_req_w] = accumulated_error + sum_error
                        old_requests[fix_req_w] = accumulated_error + req_error
                        new_requests.add(fix_req_w)
                
                #fix_req_s = split_generator(req.lower(), self.language_model)
                res = split_generator_complex(req, self.language_model)
                if res:
                    fix_req_s = res[0]
                    fix_list = res[1]
                    if fix_req_s not in old_requests:
                        req_error = self.clf(fix_list, self.language_model)
                        accumulated_errors[fix_req_s] = accumulated_error + 1.0
                        old_requests[fix_req_s] = accumulated_error + req_error
                        new_requests.add(fix_req_s)
                
                #complex join
                fix_req_j, fix_list_j = join_generator(req, tokens, self.language_model)
                if fix_req_j not in old_requests:
                    req_error = self.clf(fix_list_j, self.language_model)
                    accumulated_errors[fix_req_j] = accumulated_error + 1.0
                    old_requests[fix_req_j] = accumulated_error + req_error
                    new_requests.add(fix_req_j)

                fix_req_kl = keyboard_layout_generator(req)
                if fix_req_kl not in old_requests:
                    fix_tokens = preprocess_req(fix_req_kl)
                    fix_list = [Candidate(t.token.lower(), 0, 0) 
                                for t in fix_tokens if def_is_estimated_token(t)]
                    req_error = self.clf(fix_list, self.language_model)
                    accumulated_errors[fix_req_kl] = accumulated_error + len(fix_req_kl)
                    old_requests[fix_req_kl] = accumulated_error + req_error
                    new_requests.add(fix_req_kl)
            requests = new_requests
        fix_req = min(old_requests.items(), key = lambda item: item[1])[0]
        return fix_req

if __name__ == '__main__':
    try:
        print("1", file=sys.stderr)
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

        while True:
            try:
                query = input()
            except EOFError:
                break
            try:
                result = spellchecker.safe_correction(query, max_candidates=5, iterations=2)
                print(result)
            except:
                print(query)
    except RuntimeError as err:
        print(err, file=sys.stderr)
    except:
        print("Unknown error", file=sys.stderr)



