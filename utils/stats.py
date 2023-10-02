import numpy as np


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)

    deviation_form_med = np.array(data) - median

    mad = np.median(np.abs(deviation_form_med))
    mod_zscore = deviation_form_med / (consistency_correction * mad)

    return mod_zscore, mad
