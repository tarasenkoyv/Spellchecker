from dataclasses import dataclass, field
from typing import Any
from collections import deque, namedtuple
import string
from functools import reduce
import operator
from heapq import heappush, heappop
import re
import util

@dataclass(order=True)
class Transition:
    node: Any=field(compare=False)
    weight : float
    prefix: Any=field(compare=False)
    result: Any=field(compare=False)

@dataclass(order=True)
class CacheCandidate:
    def __init__(self, word, f_lm, f_fix):
        self.word = word
        self.f_lm = f_lm
        self.f_fix = f_fix
        self.weight = (-1) * (f_lm + f_fix)
    word: Any=field(compare=False)
    f_lm : Any=field(compare=False)
    f_fix : Any=field(compare=False)
    weight : float

@dataclass(order=True)
class Candidate:
    def __init__(self, word, lm_weight, error_weight):
        self.word = word
        self.lm_weight = lm_weight
        self.error_weight = error_weight
        self.weight = 1.7 * lm_weight + error_weight
    word: Any=field(compare=False)
    lm_weight : Any=field(compare=False)
    error_weight : Any=field(compare=False)
    weight : float

class Node:
    def __init__(self, value=None, lm_weight=None, end=False):
        self.end = end
        self.value = value
        self.lm_weight = lm_weight
        self.word = None
        self.children = {}
        self.max_candidates = 1

class Trie:
    def __init__(self, error_model, language_model):
        self.__len = 0
        self._root = Node()

        self.error_model = error_model
        self.language_model = language_model

        self.rus_letters = list("йцукенгшщзхъфывапролджэёячсмитьбю")
        self.eng_letters = list("qwertyuiop[]asdfghjkl;'\zxcvbnm,.")

        # russian symbols: е, о, а, с, у; ukrainian symbols: i
        self.similar_symbols = {'i': 1110, 'e': 1077, 'o': 1086, 'a': 1072, 'c': 1089, 'y': 1091, 'p': 1088}
        self.limit_weight = 12
        self.max_queue_size = 100_000
        self.max_iters = 100_000

    def add(self, word):
        node = self._root
        for part in word:
            next_node = node.children.get(part)
            if next_node is None:
                node.children[part] = Node(part)
                node = node.children[part]
            else:
                node = next_node

        if not node.end:
            self.__len += 1
            node.end = True
            node.word = word
            node.lm_weight = self.language_model.unigram_weights[word]

    def add_candidate(self, new_cand, candidates):
        word = new_cand.word
        if word in candidates:
             candidates[word].error_weight = \
                 min(new_cand.error_weight, candidates[word].error_weight)
        else:
            if len(candidates) <= self.max_candidates:
                candidates[word] = new_cand
            else:
                pass
                #cand_with_max_weight = max(candidates.values())
                #if cand_with_max_weight.weight > new_cand.weight:
                #    candidates[word] = new_cand
                #    del candidates[cand_with_max_weight.word]
        
    def find_candidates(self, prefix, max_candidates=5, limit_weight=8):
        if len(prefix) >= 5:
            self.limit_weight = 14
        else:
            self.limit_weight = limit_weight

        self.max_candidates = max_candidates
        queue = []
        candidates = {}
        heappush(queue, Transition(self._root, 0, prefix, ''))
        iter = 0
        while len(queue) > 0 and iter < self.max_iters:
            iter += 1
            curr_transition = heappop(queue)
            # prefix is processed
            if len(curr_transition.prefix) == 0:
                if curr_transition.node.end:
                    new_word = curr_transition.result
                    new_cand = Candidate(new_word, self.language_model.unigram_weights[new_word], curr_transition.weight)
                    self.add_candidate(new_cand, candidates)

            prefix_letter = curr_transition.prefix[:1]
            curr_weight = curr_transition.weight
            for trie_letter, next_node  in curr_transition.node.children.items():
                if trie_letter in (self.rus_letters + self.eng_letters):
                    if trie_letter == prefix_letter:
                        # add transition with null weight
                        heappush(queue, Transition(next_node, curr_weight,  curr_transition.prefix[1:], 
                                                   curr_transition.result + trie_letter))
                        # add transition with duplication prefix_letter
                        if prefix_letter in self.error_model.weights['']:
                            additional_weight = self.error_model.weights[''][prefix_letter]
                            if self.__transition_can_be_added(curr_weight + additional_weight, curr_transition):
                                heappush(queue, Transition(next_node, curr_weight + additional_weight, 
                                                           curr_transition.prefix, curr_transition.result + prefix_letter))
                    else:
                        if trie_letter in self.error_model.weights[prefix_letter]:
                            # add transition with replacing prefix_letter -> trie_letter
                            additional_weight = self.error_model.weights[prefix_letter][trie_letter]
                            # similar symbol
                            if prefix_letter != '' and trie_letter in self.similar_symbols \
                                and self.similar_symbols[trie_letter] == ord(prefix_letter):
                                additional_weight = 0.5
                            if self.__transition_can_be_added(curr_weight + additional_weight, curr_transition):
                                heappush(queue, Transition(next_node, curr_weight + additional_weight, 
                                                           curr_transition.prefix[1:], 
                                                           curr_transition.result + trie_letter))
                        # add transition with insert miss letters
                        if trie_letter in self.error_model.weights['']:
                            additional_weight = self.error_model.weights[''][trie_letter]
                            if self.__transition_can_be_added(curr_weight + additional_weight, curr_transition):
                                transition = Transition(next_node, curr_weight + additional_weight,
                                                        curr_transition.prefix,
                                                        curr_transition.result + trie_letter)
                                heappush(queue, transition)

                    # add transition with transposition (df -> fd)
                    if len(curr_transition.prefix) > 1:
                        if trie_letter == curr_transition.prefix[1] and prefix_letter in next_node.children \
                            and trie_letter != prefix_letter:
                            additional_weight = 4.0 # penalty for transposition
                            if self.__transition_can_be_added(curr_weight + additional_weight, curr_transition):
                                transition = Transition(next_node.children[prefix_letter], curr_weight + additional_weight, 
                                                        curr_transition.prefix[2:], 
                                                        curr_transition.result + trie_letter + prefix_letter)
                                heappush(queue, transition)
                    
                    #self.__add_transition_translit(queue, candidates, currcurr_transition, next_node)  
                                
            if '' in self.error_model.weights[prefix_letter]:
                # add transition with deletion current letter
                additional_weight = self.error_model.weights[prefix_letter]['']
                if self.__transition_can_be_added(curr_weight + additional_weight, curr_transition):
                    transition = Transition(curr_transition.node, curr_weight + additional_weight, 
                                            curr_transition.prefix[1:], curr_transition.result)
                    heappush(queue, transition)
            continue

        return candidates

    def __transition_can_be_added(self, weight, curr_transition):
        return weight < self.limit_weight
        #return weight < self.limit_weight and len(candidates) < self.max_candidates \
        #    and len(queue) < self.max_queue_size

    def __add_transition_translit(self, queue, candidates, curr_transition, next_node):
        prefix_letter = curr_transition.prefix[0]
        curr_weight = curr_transition.weight
        trie_letter = next_node.value
        # one translit error: rus -> eng
        rus_to_eng_symbol_error = prefix_letter in self.rus_letters \
            and trie_letter == self.eng_letters[self.rus_letters.index(prefix_letter)]
        # one translit error: eng -> rus
        eng_to_rus_symbol_error = prefix_letter in self.rus_letters \
            and trie_letter == self.eng_letters[self.rus_letters.index(prefix_letter)]
        if rus_to_eng_symbol_error or eng_to_rus_symbol_error:
            additional_weight = 0.1
            if self.__transition_can_be_added(curr_weight + additional_weight, candidates):
                heappush(queue, Transition(child, curr_weight + additional_weight, 
                                           curr_transition.prefix[1:], curr_transition.result + trie_letter))

    def _find(self, key):
        node = self._root
        for part in key:
            node = node.children.get(part)
            if node is None:
                break
        return node

    def __contains__(self, key):
        node = self._find(key)
        return node is not None and node.end

    def __len__(self):
        return self.__len

    def build(self):
        correct_words = list(self.language_model.unigram_stat.keys())
        for word in correct_words:
            self.add(word)


