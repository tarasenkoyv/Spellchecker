from collections import Counter, defaultdict
import functools
import re
import numpy as np

class LanguageModel:
    def __init__(self):
        self.alpha = 1e-05
        self.unigram_stat = defaultdict(int)
        self.bigram_stat = defaultdict(functools.partial(defaultdict, int))
        self.bigram_weights = defaultdict(functools.partial(defaultdict, float))
        self.unigram_weights = defaultdict(functools.partial(defaultdict, float))
        self.unigram_def_value = None
        self.bigram_dict = {}

    def update_unigram_stat(self, word):
        self.unigram_stat[word] += 1
    
    def update_bigram_stat(self, bigram):
        words = bigram.split('|')
        self.bigram_stat[words[0]][words[1]] += 1
    
    def build_from_file(self, filename):
        idx_line = 1
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.lower()
                if '\t' in line:
                    line = line[(line.index('\t') + 1):]
                words = re.findall(r'\w+', line)
                for i, word in enumerate(words):
                    self.unigram_stat[word] += 1
                    if (i + 1) <= (len(words) - 1):
                        self.bigram_stat[word][words[i + 1]] += 1
                idx_line += 1
        
        #self.build_bigram_dict()
        self.calc_weights()

    def build_bigram_dict(self):
        all_entries = np.sum(list(self.unigram_stat.values()))
        for w1, w1_dict in self.bigram_stat.items():
            for w2, entries in w1_dict.items():
                p_independent = (self.unigram_stat[w1] * self.unigram_stat[w2]) / (all_entries)
                p_together = entries
                if p_independent < p_together:
                    self.bigram_dict[(w1, w2)] = p_together / all_entries

    def calc_weights(self):
        all_entries = np.sum(list(self.unigram_stat.values()))
        self.unigram_def_value = -np.log(self.alpha / (all_entries + self.alpha * len(self.unigram_stat)))
        self.unigram_weights = defaultdict(functools.partial(float, self.unigram_def_value))
        for word, cnt in self.unigram_stat.items():
            self.unigram_weights[word] = -np.log(cnt / (all_entries + self.alpha))
        
        for word1, word2_dict in self.bigram_stat.items():
            for word2, cnt in word2_dict.items():
                self.bigram_weights[word1][word2] = - np.log(cnt / self.unigram_stat[word1])
