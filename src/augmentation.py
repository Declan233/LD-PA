import numpy as np

def addGaussianNoise(traces, max_noise_level):
    '''Add GaussianNoise. 
    input: 
    - traces: input traces to be desynced, shape (nb_trs, len_trs).
    - max_noise_level: maximun noise level.
    output: 
    - nl: moise level list indicating the nosie added to each tarce.
    - output_traces: desynced traces, shape (nb_trs, len_trs).
    '''
    if max_noise_level == 0:
        return traces
    else:
        nb_trs, len_trs = traces.shape
        output_traces = np.zeros_like(traces)
        nl = np.random.rand(nb_trs) * max_noise_level
        for ti in range(nb_trs):
            profile_trace = traces[ti]
            noise = np.random.normal(0, nl[ti], size=len_trs)
            output_traces[ti] = profile_trace + noise
        return output_traces
