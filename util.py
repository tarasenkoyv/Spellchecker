import nltk_util
import pickle
import operator

def save_obj(obj, name):
    p = pickle.Pickler(open(name + '.pkl', 'wb'))
    p.fast = True
    p.dump(obj)

def load_obj(name):                                                            
    with open(name + '.pkl', 'rb') as f:                                        
        return pickle.load(f)

def evaluate_pair_words_nll(curr_words, next_words, language_model):
    best_pair = []
    l_best_pair = float("inf")
    if len(next_words) > 1 or len(curr_words) > 1:
        for curr_word in curr_words:
            for next_word in next_words:
                pair = [curr_word, next_word]
                l_pair = evaluate_words_nll([curr_word, next_word], language_model)
                if l_pair < l_best_pair:
                    best_pair = pair
                    l_best_pair = l_pair
    else:
        best_pair = [curr_words[0], next_words[0]]
    return best_pair

def evaluate_pair_candidates_nll(curr, next, language_model):
    best_pair = []
    l_best_pair = float("inf")
    if len(next) > 1 or len(curr) > 1:
        for c in curr:
            for n in next:
                l_pair = evaluate_words_nll([c.word, n.word], language_model)
                if l_pair < l_best_pair:
                    best_pair = [c, n]
                    l_best_pair = l_pair
    else:
        best_pair = [curr[0], next[0]]
    return best_pair

def evaluate_words_nll(words, language_model, smoothing=True):
    words = [w.lower() for w in words]
    negative_log_p0 = 1_000
    if len(words) == 0: return negative_log_p0
    l_req = 0
    if language_model.unigram_stat[words[0]] > 0:
        l_req += language_model.unigram_weights[words[0]]
    else:
        l_req += language_model.unigram_def_value if smoothing else negative_log_p0
    for i, word in enumerate(words[1:]):
        if language_model.unigram_stat[words[i]] == 0:
            l_req += language_model.unigram_def_value if smoothing else negative_log_p0
        else:
            if language_model.bigram_weights[words[i]][word] == 0:
                if language_model.unigram_stat[word] > 0:
                    l_req += language_model.unigram_weights[word]
                else:
                    l_req += language_model.unigram_def_value if smoothing else negative_log_p0
            else:
                l_req += language_model.bigram_weights[words[i]][word]
    return l_req

def evaluate_req_nll(request, language_model, smoothing=True):
    words = request.split()
    return evaluate_words_nll(words, language_model, smoothing)

def edit_matrix(s1, s2, substitution_cost=1, transpositions=False):
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = nltk_util._edit_dist_init(len1 + 1, len2 + 1)

    # retrieve alphabet
    sigma = set()
    sigma.update(s1)
    sigma.update(s2)

    # set up table to remember positions of last seen occurrence in s1
    last_left_t = nltk_util._last_left_t_init(sigma)

    # iterate over the array
    for i in range(len1):
        last_right = 0
        for j in range(len2):
            last_left = last_left_t[s2[j]]
            nltk_util._edit_dist_step(
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
    return lev
