from dataclasses import dataclass, field
from typing import Any
import operator
import numpy as np
import util
import re
import nltk_util
import copy
from collections import defaultdict
from trie import Candidate
import itertools
from classifiers import stat_clf

re_preprocess_req1 = r"((?:\s+)|(\S+))"
re_preprocess_req2 = r"((\w+)|(?:\W+))"
re_email = r"^([\w\.-]+)@([\w\.-]+)(\.[\w\.]+)$"
#re_url = r"^(https?:\/\/)?(?:[-\w]+\.)?([-\w]+)\.\w+(?:\.\w+)?\/?.*$"
re_url = r"^(https?:\/\/)(?:[-\w]+\.)?([-\w]+)\.\w+(?:\.\w+)?\/?.*$"
re_digit = r"^(\d+)$"
re_j1 = r'^(?:(\w) (\d\d site:\.\w{2,4}))$'

@dataclass(order=True)
class CandidateList:
    def __init__(self, candidates, language_model):
        self.candidates = candidates
        self.weight = stat_clf(candidates, language_model)

    def add(self, cand, language_model):
        self.candidates.append(cand)
        self.weight = stat_clf(self.candidates, language_model)

    candidates: Any=field(compare=False)
    weight : float

class Token:
    def __init__(self, token, need_correct, is_delim=False, is_stop_word=False, is_one_symbol=False):
        self.token = token
        self.need_correct = need_correct
        self.is_delim = is_delim
        self.is_spec = False
        self.is_digit = False
        self.fix_token = self.token
        self.fix_error = 0
        if self.need_correct:
            self.is_first_upper = token[0].isupper()
            self.is_all_upper = token.isupper()
        else:
            self.is_first_upper = False
            self.is_all_upper = False

        self.fix_words = []
        self.is_stop_word = is_stop_word
        self.is_one_symbol = is_one_symbol

    def __repr__(self):
        rep = 'Token(' + self.token + ', ' + str(self.is_delim) + ', ' \
            + str(self.need_correct) + ', is_first_upper=' + str(self.is_first_upper) \
            +')'
        return rep

def def_is_corrected(token):
    return token.is_stop_word or token.is_one_symbol or token.need_correct \
        or token.is_digit

def def_is_estimated_token(token):
    return not token.is_delim
    #return token.is_stop_word or token.is_one_symbol or token.need_correct \
    #    or token.is_digit

def def_is_spec_token(token):
    res = (re.search(re_email, token) or re.search(re_url, token))
    return res

def def_is_digit_token(token):
    return re.search(re_digit, token)

def def_is_spec_join_token(req):
    res = re.findall(re_j1, req)
    if res:
        fix_req = res[0][0] + res[0][1]
        return fix_req
    else:
        return ''

def def_is_stop_word(token):
    res = token in (nltk_util.stop_words_en + nltk_util.stop_words_ru)
    return res

def preprocess_req(req, second_stage=True):
    l1 = []
    for t in re.findall(re_preprocess_req1, req):
        l1.append(Token(t[0], t[1] != '', t[1] == ''))
    if not second_stage: return l1
    l2= []
    for t in l1:
        if t.is_delim:
            l2.append(t)
            continue
        # second check
        if def_is_spec_token(t.token):
            t.is_stop_word = False
            t.need_correct = False
            t.is_spec = True
            l2.append(t)
            continue
        if def_is_digit_token(t.token):
            t.need_correct = False
            t.is_digit = True
            l2.append(t)
            continue
        if def_is_stop_word(t.token):
            t.is_stop_word = True
            t.need_correct = False
            l2.append(t)
            continue

        for t2 in re.findall(re_preprocess_req2, t.token):
            is_delim = (t2[1] == '')
            is_stop_word = not is_delim and def_is_stop_word(t2[0])
            is_spec_word = not is_delim and def_is_spec_token(t2[0])
            is_digit_word = not is_delim and def_is_digit_token(t2[0])
            is_one_symbol = (len(t2[0]) == 1) and not is_spec_word \
                and not is_delim and not is_digit_word
            need_correct =  not is_delim and not is_stop_word \
                and not is_spec_word and not is_one_symbol and not is_digit_word
            new_token = Token(t2[0], need_correct, is_delim, is_stop_word, is_one_symbol)
            new_token.is_digit = is_digit_word
            new_token.is_spec_word = is_spec_word
            l2.append(new_token)
    return l2

def reconstruct_req(tokens, fix_dict=None):
    req = ''
    for i, t in enumerate(tokens):
        if not t.need_correct:
            req += t.token
        else:
            fix_word = fix_dict[i] if fix_dict else t.token
            if t.is_all_upper:
                req += fix_word.upper()
            elif t.token.lower() == fix_word:
                req += t.token
            elif len(t.token) == len(fix_word):
                for j, c in enumerate(t.token):
                    fix_c = fix_word[j]
                    req += fix_c.upper() if c.isupper() else fix_c
            elif t.is_first_upper:
                req += fix_word.capitalize()
            else:
                req += fix_word
    return req

def word_generator(tokens, language_model, trie, max_candidates=5):
    """
    Fixing typos in query words
    """
    fix_words_l = []
    tokens_fix_indices = []
    for i, token in enumerate(tokens):
        orig_word = token.token.lower()
        if token.need_correct:
            candidates = trie.find_candidates(orig_word, max_candidates)
            fix_words = sorted([c for c in candidates.values() if language_model.unigram_stat[c.word] > 0])
            fix_words = fix_words if len(fix_words) > 0 else [Candidate(orig_word, 
                                                                        language_model.unigram_weights[orig_word],
                                                                        0)]
            fix_words_l.append(fix_words)
            tokens_fix_indices.append(i)
        elif def_is_estimated_token(token):
            fix_words_l.append([Candidate(orig_word, language_model.unigram_weights[orig_word], 0)])
            tokens_fix_indices.append(i)

    res = []
    if len(fix_words_l) == 1:
        fix_list = [fix_words_l[0][0]]
        fix_dict = {token_idx: fix_words_l[cand_idx][0].word
                    for cand_idx, token_idx in enumerate(tokens_fix_indices) if tokens[token_idx].need_correct}
        fix_req_text = reconstruct_req(tokens, fix_dict)
        res.append((fix_req_text, fix_list))
        return res

    # top 5
    res_cl = []
    res_cl.extend([CandidateList([c], language_model) for c in fix_words_l[0][:5]])
    for i, next_list in enumerate(fix_words_l[1:]):
        next_list = next_list[:10]
        res_cl_new = []
        for cl in res_cl:
            for curr_cand in next_list:
                cl_new = copy.deepcopy(cl)
                cl_new.add(curr_cand, language_model)
                res_cl_new.append(cl_new)
        res_cl = sorted(res_cl_new)[:3]
    for cl in res_cl:
        fix_dict = {token_idx: cl.candidates[cand_idx].word
                    for cand_idx, token_idx in enumerate(tokens_fix_indices) if tokens[token_idx].need_correct}
        res.append((reconstruct_req(tokens, fix_dict), cl.candidates))
    
    return res

def keyboard_layout_generator(request):
    rus_letters = 'йцукенгшщзхъфывапролджэёячсмитьбю'
    rus_letters = list(rus_letters + rus_letters.upper())
    eng_letters = "qwertyuiop[]asdfghjkl;'\zxcvbnm,."
    eng_letters = list(eng_letters + eng_letters.upper())
    fix_request = ""
    for letter in request:
        if letter in rus_letters:
            fix_request += eng_letters[rus_letters.index(letter)]
        elif letter in eng_letters:
            fix_request += rus_letters[eng_letters.index(letter)]
        else:
            fix_request += letter
    return fix_request

def def_can_join(token):
    #return token.need_correct or token.is_one_symbol or token.is_stop_word
    return not token.is_delim

def join_generator(request, tokens, language_model):
    indices = [i for i, t in enumerate(tokens) if t.is_delim]
    fix_request = request
    fix_tokens = tokens
    fix_cl = [Candidate(t.token, 0, 0) for t in fix_tokens if def_is_estimated_token(t)]
    if len(fix_cl) == 0: return (fix_request, fix_cl)

    fix_request_l = stat_clf(fix_cl, language_model)
    join_cnt = 0 # number of join made
    for idx in indices:
        can_join = def_can_join(fix_tokens[idx-1-join_cnt]) \
            and (idx+1-join_cnt) <= (len(fix_tokens) - 1) \
            and def_can_join(fix_tokens[idx+1-join_cnt])
        if can_join:
            joined_token = Token(fix_tokens[idx-1-join_cnt].token + fix_tokens[idx+1-join_cnt].token, True)
            candidate_tokens = fix_tokens[:idx-1-join_cnt] + [joined_token] + fix_tokens[idx - join_cnt + 2:]
            candidate_cl = [Candidate(t.token, 0, 0) for t in candidate_tokens if def_is_estimated_token(t)]
            candidate_l = stat_clf(candidate_cl, language_model)
            if  candidate_l < fix_request_l:
                fix_tokens = candidate_tokens
                fix_request_l = candidate_l
                fix_cl = candidate_cl
                join_cnt += 2
    return (reconstruct_req(fix_tokens), fix_cl)

def join_generator_simple(request, language_model):
    indices = [i for i, v in enumerate(request) if v == ' ']
    fix_request = request
    fix_request_l = util.evaluate_req_nll(fix_request, language_model)
    join_cnt = 0 # number of join made
    for idx in indices:
        candidate = fix_request[:idx - join_cnt] + fix_request[idx - join_cnt + 1:]
        candidate_l = util.evaluate_req_nll(candidate, language_model, smoothing=False)
        if  candidate_l < fix_request_l:
            fix_request = candidate
            fix_request_l = candidate_l
            join_cnt += 1
    return fix_request

def def_can_split(token):
    return token.need_correct

def split_generator_complex(req, language_model):
    tokens = preprocess_req(req, second_stage=True)
    new_tokens = []
    new_fix_cl = []
    is_split =False
    for i, token in enumerate(tokens):
        if not token.is_delim and not token.is_digit:
            fix_tokens, fix_cl, is_token_split = split_generator(token, language_model)
            if is_token_split:
                is_split = True
                new_tokens.extend(fix_tokens)
                new_fix_cl.extend(fix_cl)
            else:
                new_tokens.append(token)
                new_fix_cl.append(Candidate(token.token, 0, 0))
        else:
            new_tokens.append(token)
    if is_split:
        return (reconstruct_req(new_tokens), new_fix_cl)
    else:
        return False

def split_generator(token, language_model):
    is_split = False
    text_to_split = token.token
    indices = np.array([i for i, _ in enumerate(text_to_split[1:]) if text_to_split[i] != ' ']) + 1
    fix_tokens = [token]
    fix_cl = [Candidate(text_to_split, 0, 0)]
    fix_request_l = stat_clf(fix_cl, language_model)
    for idx in indices:
        candidate = text_to_split[:idx] + ' ' + text_to_split[idx:]
        candidate_tokens = preprocess_req(candidate)
        candidate_cl = [Candidate(t.token, 0, 0) for t in candidate_tokens if not t.is_delim]
        #error_weight = sum([(-2) * int(t.is_spec) for t in candidate_tokens])
        candidate_l = stat_clf(candidate_cl, language_model)
        #candidate_l += error_weight
        if candidate_l < fix_request_l:
            is_split = True
            fix_tokens = candidate_tokens
            fix_cl = candidate_cl
    if is_split:
        for t in fix_tokens:
            t.need_correct = not t.is_delim
    return (fix_tokens, fix_cl, is_split)
