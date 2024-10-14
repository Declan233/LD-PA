import numpy as np
from numba import njit


@njit
def fast_key_rank(nt, n_ge, probabilities_kg_all_traces, nb_guesses, correct_key, key_ranking_sum, success_rate_sum):
    r = np.random.choice(nt, n_ge, replace=False) # randomly select n_ge index out of nt.
    probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
    key_probabilities = np.zeros(nb_guesses)
    kr_count = 0
    for index in range(n_ge):
        key_probabilities += probabilities_kg_all_traces_shuffled[index]
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
        key_ranking_good_key = list(key_probabilities_sorted).index(correct_key) + 1
        key_ranking_sum[kr_count] += key_ranking_good_key
        if key_ranking_good_key == 1:
            success_rate_sum[kr_count] += 1
        kr_count += 1


def sca_metrics(output_probabilities, n_ge, label_key_guess, correct_key):
    key_ranking_sum = np.zeros(n_ge)
    success_rate_sum = np.zeros(n_ge)

    nt = len(output_probabilities)
    nb_guesses = len(label_key_guess)
    probabilities_kg_all_traces = np.zeros((nt, nb_guesses))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in label_key_guess[:]])  # array with 256 leakage values (1 per key guess)
        ]

    for _ in range(100):
        fast_key_rank(nt, n_ge, probabilities_kg_all_traces, nb_guesses, correct_key, key_ranking_sum, success_rate_sum)

    guessing_entropy = key_ranking_sum / 100
    success_rate = success_rate_sum / 100
    if guessing_entropy[n_ge - 1] < 2:
        result_number_of_traces_ge_1 = n_ge - np.argmax(guessing_entropy[::-1] > 2)
    else:
        result_number_of_traces_ge_1 = n_ge

    return guessing_entropy, success_rate, result_number_of_traces_ge_1


def SNR_fit(traces, nb_cls, iv):
    _, nb_iv = iv.shape
    snr_val = []
    for i in range(nb_iv):
        snr_val.append(SNR_simple(traces, iv[:,i], nb_cls))
    return np.array(snr_val)


def SNR_simple(traces, share, nb_cls):  # calculate SNR using IV
    """
    according to <Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database>
    P11 bottom, in paper
    :param traces: power consumption wave (n, m)
    :param iv: iv for classification (n, )
    :param nb_cls: number of classes
    :return:
    """
    traces = np.array(traces)
    share = np.array(share)
    assert share.ndim==1 and traces.ndim==2 and len(traces)==len(share), \
          print(f"Illegal dimension for share[shape:(n,)] or traces[shape:(n,m)]. Input shape {share.shape} for share and {traces.shape} for traces.")
    # classify all traces using share value
    iv_idxs = [[] for _ in range(nb_cls)]
    for i in range(len(traces)):
        iv_idxs[share[i]].append(i)
    # calculate mean trace of each class
    # print(len(iv_idxs), len(iv_idxs[0]), len(iv_idxs[1]))
    mean_trace = []
    var_trace = []
    for each in iv_idxs:
        if len(each) != 0:
            mean_trace.append(np.mean(traces[each, :], axis=0))
            var_trace.append(np.var(traces[each, :], axis=0))
    # print(mean_trace, var_trace)
    # print(np.var(mean_trace, axis=0), np.mean(var_trace, axis=0))
    return np.var(mean_trace, axis=0) / np.mean(var_trace, axis=0)
