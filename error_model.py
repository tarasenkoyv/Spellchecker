import util
from collections import Counter, defaultdict
import functools
import numpy as np
import operator

class ErrorModel:
    def __init__(self):
        # transposition
        #self.trans_stat = defaultdict(functools.partial(defaultdict, int))
        # insert, delete, substitution
        self.stat = defaultdict(functools.partial(defaultdict, int))
        self.all_errors = 0
        pass

    def update_stat(self, s1, s2):
        lev = util.edit_matrix(s1, s2, substitution_cost=1, transpositions=False)
        i, j = len(lev) - 1, len(lev[0]) - 1

        while (i, j) != (0, 0):
            directions = [
                (i - 1, j - 1),  # substitution (s[i] to s[j])
                (i, j - 1),  # insert s2[j]
                (i - 1, j),  # delete s1[i]
            ]

            direction_costs = (
                (oper, lev[i][j] if (i >= 0 and j >= 0) else float("inf"))
                 for oper, (i, j) in enumerate(directions))
            
            oper, dist = min(direction_costs, key=operator.itemgetter(1))
            
            if dist != lev[i][j]:
                if oper == 0: # substitution
                    self.stat[s1[i-1]][s2[j-1]] += 1
                    self.all_errors += 1
                elif oper == 1: # insert s2[j]
                    self.stat[''][s2[j-1]] += 1
                    self.all_errors += 1
                else: # delete s1[i]
                    self.stat[s1[i-1]][''] += 1
                    self.all_errors += 1
                
            (i, j) = directions[oper];

    def calc_weights(self):
        self.weights = defaultdict(functools.partial(defaultdict, float))
        for l1, dict_values in self.stat.items():
            #l1_overall = np.sum(list(dict_values.values())); 
            for l2, l2_cnt in dict_values.items():
                self.weights[l1][l2] = -np.log(l2_cnt / self.all_errors)
                #self.weights[l1][l2] = -np.log(l2_cnt / l1_overall)

    def build_from_file(self, filename):
        orig_requests = []
        fix_requests = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if '\t' in line:
                    line = line.rstrip('\n').lower()
                    orig_requests.append(line[:(line.index('\t'))])
                    fix_requests.append(line[(line.index('\t') + 1):])

        for orig, fix in zip(orig_requests, fix_requests):
            self.update_stat(orig, fix)

        self.calc_weights()