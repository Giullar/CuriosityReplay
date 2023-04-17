import os
import numpy as np
import pandas as pd
from ast import literal_eval

# PLOT AVERAGE EVALUATION AT TIMESTEPS 300K AND 400K
def compute_eval_scores_tms(dir_env_dfs, models_to_plot, selected_agent, timesteps=[300_000, 400_000]):
    eval_scores = {}
    for model in models_to_plot:
        # Load the dataframe
        df = pd.read_csv(os.path.join(dir_env_dfs, "eval_returns_"+selected_agent+"_"+model+".csv"), converters={"y": literal_eval})
        eval_scores[model] = {}
        for tms in timesteps:
            # Get the last correct rows of the dataframe
            df_filtered = df[df["x"]==tms]
            rows = df_filtered["y"].values.tolist()
            # compute mean and std
            mean = np.mean(rows, axis=1)
            std = np.std(rows, axis=1)
            eval_scores[model][tms] = {"mean":mean[0], "std":std[0]}
    return eval_scores

def print_eval_scores(eval_scores):
    # Print the eval scores
    for model in eval_scores:
        print("STATISTICS FOR "+str(model)+":")
        model_tms = eval_scores[model]
        for tms in model_tms:
            print("TMS == "+str(tms)+":")
            model_stats = model_tms[tms]
            for stat in model_stats:
                print(str(stat)+" : "+str(model_stats[stat]))
            print()
        print()
        print()


