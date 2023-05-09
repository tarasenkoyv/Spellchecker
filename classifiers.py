import util

def stat_clf(candidates, language_model):
    words = [c.word.lower() for c in candidates]
    fix_error = sum([c.error_weight for c in candidates])
    return 1.7 * util.evaluate_words_nll(words, language_model, smoothing=False) + fix_error

#def stat_clf(tokens, language_model):
#    words = [t.fix_token.lower() for t in tokens if def_is_estimated_token(t)]
#    fix_error = sum([t.fix_error for t in tokens if def_is_estimated_token(t)])
#    return util.evaluate_words_nll(words, language_model) + fix_error
