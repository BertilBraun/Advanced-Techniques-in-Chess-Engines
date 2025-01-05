from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

USAGE_FOLDER = R'C:\Users\berti\OneDrive\Desktop\zip2\zip'


def load_usage(folder_path: str) -> pd.DataFrame:
    # find all files in folder of format usage_*.csv - load them all and return them in a single dataframe

    return pd.concat(pd.read_csv(file, parse_dates=['timestamp']) for file in Path(folder_path).glob('usage_*.csv'))


def visualize():
    system_df = load_usage(USAGE_FOLDER)
    system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])

    system_df = system_df.sort_values(by='timestamp')
    # Recompute time differences
    system_df['time_diff'] = system_df['timestamp'].diff().dt.total_seconds()

    # Remove rows with negative or unusually large gaps
    system_df = system_df[system_df['time_diff'] > 0]

    ranks = system_df['rank'].unique()
    gpu_ids = system_df['gpu_id'].unique()

    axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)[1]

    # CPU Usage
    for rank in ranks:
        df_rank = system_df[system_df['rank'] == rank]
        axs[0].plot(df_rank['timestamp'], df_rank['cpu_percent'], label=f'CPU Rank {rank}')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].set_title('Process-Specific CPU Usage')
    axs[0].legend(loc='upper left')

    # RAM Usage
    for rank in ranks:
        df_rank = system_df[system_df['rank'] == rank]
        axs[1].plot(df_rank['timestamp'], df_rank['ram_usage'], label=f'RAM Rank {rank}')
    axs[1].set_ylabel('RAM Usage (MB)')
    axs[1].set_title('Process-Specific RAM Usage')
    axs[1].legend(loc='upper left')

    # GPU Load
    for gpu in gpu_ids:
        df_gpu = system_df[system_df['gpu_id'] == gpu]
        axs[2].plot(df_gpu['timestamp'], df_gpu['gpu_load'], label=f'GPU {gpu} Load')
    axs[2].set_ylabel('GPU Load (%)')
    axs[2].set_title('GPU Load')
    axs[2].legend(loc='upper left')

    # GPU VRAM Usage
    for gpu in gpu_ids:
        df_gpu = system_df[system_df['gpu_id'] == gpu]
        axs[3].plot(df_gpu['timestamp'], df_gpu['gpu_memory_used'], label=f'GPU {gpu} VRAM Used')
        axs[3].plot(df_gpu['timestamp'], df_gpu['gpu_memory_total'], label=f'GPU {gpu} VRAM Total')
    axs[3].set_ylabel('VRAM Usage (MB)')
    axs[3].set_title('GPU VRAM Usage')
    axs[3].legend(loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
