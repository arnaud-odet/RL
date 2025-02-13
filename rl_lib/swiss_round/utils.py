import pandas as pd
import numpy as np

def probability_tables(team_strengths, max_draw_probability):
    index = range(len(team_strengths))
    wps = []
    dps = []
    lps = []
    for ts1 in team_strengths :
        twps = []
        tdps = []
        tlps = []
        for ts2 in team_strengths :
            strength_diff = ts1 - ts2
            tmp_win_prob = 1 / (1 + np.exp(-strength_diff))
            tmp_loss_prob = 1 / (1 + np.exp(+strength_diff))
            tmp_draw_prob = max_draw_probability * np.exp(-abs(strength_diff))
            # Softmax
            win_prob = tmp_win_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)
            draw_prob = tmp_draw_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)
            loss_prob = tmp_loss_prob / (tmp_win_prob + tmp_draw_prob + tmp_loss_prob)  
            
            twps.append(win_prob)
            tdps.append(draw_prob)
            tlps.append(loss_prob)
        wps.append(twps)
        dps.append(tdps)
        lps.append(tlps)
    return pd.DataFrame(wps, index=index, columns = index), pd.DataFrame(dps, index=index, columns = index),pd.DataFrame(lps, index=index, columns = index)  

def check_probability(team_strengths, max_draw_probability):
    wp, dp, lp = probability_tables(team_strengths=team_strengths, max_draw_probability=max_draw_probability)
    df = wp+dp+lp
    values_array = df.to_numpy()
    target_array = np.full_like(values_array, 1)

    return np.allclose(values_array, target_array, rtol=10e-5, atol=10e-8)