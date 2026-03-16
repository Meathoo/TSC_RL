import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# 預設顏色與線型
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']

def smooth_data(data, window_size=5):
    """對資料進行移動平均平滑處理"""
    if window_size <= 1:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def load_metric(folder, metric_name):
    """載入指定資料夾的指標資料"""
    file_path = os.path.join(folder, f"episode_{metric_name}.npy")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None
    data = np.load(file_path)
    if data.ndim > 1:
        data = data.mean(axis=tuple(range(1, data.ndim)))
    return data

def process_and_plot(ax, folders, names, metric_info, start_idx, end_idx, smooth_window):
    """封裝繪圖邏輯，供總圖與單獨圖調用"""
    metric_name, title, ylabel = metric_info
    
    for i, (folder, name) in enumerate(zip(folders, names)):
        data = load_metric(folder, metric_name)
        if data is None: continue
        
        # 數據切片
        actual_end = end_idx if end_idx is not None else len(data)
        sliced_data = data[start_idx:actual_end]
        base_x = np.arange(start_idx, start_idx + len(sliced_data))
        
        # 平滑處理
        if smooth_window > 1 and len(sliced_data) > smooth_window:
            smoothed = smooth_data(sliced_data, smooth_window)
            offset = smooth_window // 2
            x = base_x[offset : offset + len(smoothed)]
        else:
            smoothed = sliced_data
            x = base_x
        
        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(x, smoothed, label=name, color=color, linestyle=linestyle, linewidth=1.5)
    
    ax.set_title(f"{title} ({start_idx}-{actual_end})", fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_comparison(folders, names, output_folder, smooth_window=1, figsize=(16, 12), start_idx=0, end_idx=None):
    metrics = [
        ('average_travel_time', 'Average Travel Time', 'Average Travel Time (s)'),
        ('intersection_reward', 'Intersection Reward', 'Reward'),
        ('throughput', 'Throughput', 'Throughput (vehicles)'),
        ('average_queue_length', 'Average Queue Length', 'Queue Length (vehicles)')
    ]
    
    os.makedirs(output_folder, exist_ok=True)
    range_suffix = f"_{start_idx}_to_{end_idx if end_idx else 'end'}"

    # 1. 繪製 2x2 總表
    fig_all, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()
    
    print(f"正在生成總表與個別指標圖 (範圍: {start_idx} 之後)...")
    
    for ax_idx, m_info in enumerate(metrics):
        # 繪製到總表
        process_and_plot(axes_flat[ax_idx], folders, names, m_info, start_idx, end_idx, smooth_window)
        
        # 2. 繪製並儲存「單獨」的圖
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        process_and_plot(ax_single, folders, names, m_info, start_idx, end_idx, smooth_window)
        
        single_output = os.path.join(output_folder, f"comparison_{m_info[0]}{range_suffix}.png")
        fig_single.savefig(single_output, dpi=150, bbox_inches='tight')
        plt.close(fig_single) # 釋放記憶體

    # 儲存總表
    fig_all.tight_layout()
    all_output = os.path.join(output_folder, f"comparison_all_metrics{range_suffix}.png")
    fig_all.savefig(all_output, dpi=150, bbox_inches='tight')
    plt.close(fig_all)
    
    print(f"所有圖表已儲存至: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='比較多個模型的訓練曲線 (包含總表與個別圖)')
    parser.add_argument('-f', '--folders', type=str, nargs='+', required=True, help='資料夾列表')
    parser.add_argument('-n', '--names', type=str, nargs='+', required=True, help='模型名稱列表')
    parser.add_argument('-o', '--output', type=str, default='./comparison_results', help='輸出路徑')
    parser.add_argument('--smooth', type=int, default=1, help='平滑窗口')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 12], help='總表尺寸')
    parser.add_argument('--start', type=int, default=0, help='起始 Episode')
    parser.add_argument('--end', type=int, default=None, help='結束 Episode')
    
    args = parser.parse_args()
    
    if len(args.folders) != len(args.names):
        print(f"Error: 資料夾與名稱數量不匹配")
        return

    plot_comparison(
        folders=args.folders,
        names=args.names,
        output_folder=args.output,
        smooth_window=args.smooth,
        figsize=tuple(args.figsize),
        start_idx=args.start,
        end_idx=args.end
    )

if __name__ == "__main__":
    main()

# python compare.py -f /home/meathoo/RegionLight/TSC_RL_origin/TSC_RL/record/6x6_6_6_bi_originABDQ /home/meathoo/RegionLight/TSC_RL_origin/TSC_RL/record/6x6_6_6_bi_lf  /home/meathoo/RegionLight/TSC_RL_hie2/records/6x6_6_6_bi_hie2_lw  -n ABDQ lf hie2-2 --start 0 --smooth 100 -o ./comparison_results_6x6_lw_smooth100