import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import pandas as pd
from constants import IMG_DPI


# Apply a smooting average to smooth the time series.
# The function returns both the smoothed y and x, converted into pandas dataframe objects.
def smooth_time_series(x, y, windows_size=1):
    if windows_size > 1:
        points_list_pd = pd.DataFrame(y)
        smoothed_list_pd = points_list_pd.rolling(windows_size, center=True).mean() # Apply moving average
        return x[int(windows_size/2) : -int(windows_size/2)], smoothed_list_pd[int(windows_size/2) : -int(windows_size/2)]
    else:
        return x, y

def print_results_from_dataframe_ci(dfs, labels, file_name, xlabel="Steps", ylabel="Reward", color=None):
    z = 1.96 # value of z for 95% confidence
    for df,label in zip(dfs, labels):
        # obtain x and y points
        x_s = df["x"]
        rewards_lists = df["y"].values.tolist()
        y_s = np.mean(rewards_lists, axis=1)
        # Compute the Standard Error (SE) from the stadard deviations (SD)
        # SE = SD / sqrt(N)
        n_rollouts = len(rewards_lists[0])
        standard_errors = np.divide(np.std(rewards_lists, axis=1), np.sqrt(n_rollouts))
        # Compute the confidence intervals with z * SD/sqrt(N)
        ci_s = z * standard_errors
        # Plot (x,y) points and confidence intervals
        if color is not None:
            plt.plot(x_s, y_s, label=label, color=color)
            plt.fill_between(x_s, (y_s-ci_s), (y_s+ci_s), color=color, alpha=.3)
        else:
            plt.plot(x_s, y_s, label=label)
            plt.fill_between(x_s, (y_s-ci_s), (y_s+ci_s), alpha=.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(file_name, dpi=IMG_DPI)
    plt.show()

def print_ep_rewards_from_dataframe(df, label, file_name):
    x_s = df["x"]
    y_s = df["y"]
    plt.plot(x_s, y_s, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(file_name, dpi=IMG_DPI)
    plt.show()

def print_priorities_hist_from_dataframe(dfs, labels, windows_size, file_name):
    for df,label in zip(dfs, labels):
        if windows_size > 1:
            smoothed_df = df.rolling(windows_size, on="y", center=True).mean().dropna() # Apply moving average
            x_s = smoothed_df["x"]
            y_s = smoothed_df["y"]
        else:
            x_s = df["x"]
            y_s = df["y"]
        plt.plot(x_s, y_s, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Max Priority")
    plt.legend()
    plt.savefig(file_name, dpi=IMG_DPI)
    plt.show()

# Version with confidence intervals  
def print_buffer_priorities_from_dataframe(df, file_name):
    z = 1.96 # value of z for 95% confidence
    if len(df["tms"]) > 0:
        # Make a user-defined colormap.
        cmap = mcol.LinearSegmentedColormap.from_list("TimestepsColormap",["b","r"])
        # Make a normalizer that will map the time values from
        # [start_time,end_time+1] -> [0,1].
        cnorm = mcol.Normalize(vmin=min(df["tms"]),vmax=max(df["tms"]))
        # Turn these into an object that can be used to map time values to colors and
        # can be passed to plt.colorbar().
        cpick = cm.ScalarMappable(norm=cnorm,cmap=cmap)
        cpick.set_array([])
        # Plot priorities inside the buffer at different timesteps during training
        # for each timestep
        for tms, prs in zip(df["tms"], df["prs"]):
            prs_ordered = []
            # for each rollout recorded at timestep tms
            for rollout_prs in prs:
                # Sort the priorities of the rollout and append them to the list
                prs_ordered.append(np.flip(np.sort(rollout_prs)))
            # Compute the Standard Error (SE) from the stadard deviations (SD)
            # SE = SD / sqrt(N)
            n_rollouts = len(prs_ordered)
            standard_errors = np.divide(np.std(prs_ordered, axis=0), np.sqrt(n_rollouts))
            # Compute the confidence intervals with z * SD/sqrt(N)
            ci_s = z * standard_errors
            y_s = np.mean(prs_ordered, axis=0)
            # Plot
            plt.plot(y_s, label=tms, color=cpick.to_rgba(tms))
            #plt.fill_between(range(0, len(y_s)), np.maximum((y_s-ci_s),0), np.minimum((y_s+ci_s),1), color=cpick.to_rgba(tms), alpha=.3)
            plt.fill_between(range(0, len(y_s)), (y_s-ci_s), (y_s+ci_s), color=cpick.to_rgba(tms), alpha=.3)
        plt.xlabel("Transitions")
        plt.ylabel("Priority")
        #plt.legend()
        plt.colorbar(cpick,label="Training Timsteps")
        plt.savefig(file_name, dpi=IMG_DPI)
        plt.show()

def format_plots_path_name(dir="", env_name="", suffix=None, ext=""):
    if suffix is not None:
        return os.path.join(dir, env_name.split("/")[-1] + "-"+ suffix + ext)
    else:
        return os.path.join(dir, env_name.split("/")[-1] + ext)





