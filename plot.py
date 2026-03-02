import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_curve(data, title, ylabel, figure_file):
    # 如果資料不是一維，則壓成一維
    if data.ndim > 1:
        data = data.mean(axis=tuple(range(1, data.ndim)))
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="資料夾路徑")
    args = parser.parse_args()

    folder = args.folder

    # 1. average travel time
    avg_travel_time_path = os.path.join(folder, "episode_average_travel_time.npy")
    avg_travel_time = np.load(avg_travel_time_path)
    plot_curve(
        avg_travel_time,
        "Episode Average Travel Time",
        "Average Travel Time",
        os.path.join(folder, "plot_average_travel_time.png")
    )

    # 2. intersection reward
    intersection_reward_path = os.path.join(folder, "episode_intersection_reward.npy")
    intersection_reward = np.load(intersection_reward_path)
    plot_curve(
        intersection_reward,
        "Episode Intersection Reward",
        "Intersection Reward",
        os.path.join(folder, "plot_intersection_reward.png")
    )

    # 3. throughput
    throughput_path = os.path.join(folder, "episode_throughput.npy")
    throughput = np.load(throughput_path)
    plot_curve(
        throughput,
        "Episode Throughput",
        "Throughput",
        os.path.join(folder, "plot_throughput.png")
    )

    # 4. average queue length
    queue_length_path = os.path.join(folder, "episode_average_queue_length.npy")
    queue_length = np.load(queue_length_path)
    plot_curve(
        queue_length,
        "Episode Average Queue Length",
        "Average Queue Length",
        os.path.join(folder, "plot_average_queue_length.png")
    )

# usage python plot.py ./your_folder
# example python plot.py records/Hangzhou_4_4_real_09_14_23_05_42_ABDQ