import pandas as pd
import matplotlib.pyplot as plt


def visualize(usage_file='usage.csv'):
    system_df = pd.read_csv(usage_file, parse_dates=['timestamp'])
    system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])

    ranks = system_df['rank'].unique()
    gpu_ids = system_df['gpu_id'].unique()

    axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)[1]

    # CPU Usage
    for rank in ranks:
        df_rank = system_df[system_df['rank'] == rank]
        axs[0].plot(df_rank['timestamp'], df_rank['cpu_percent'], label=f'CPU Rank {rank}')
    axs[0].plot(
        system_df['timestamp'],
        system_df.groupby('timestamp')['cpu_percent'].mean(),
        label='Average CPU',
        color='black',
        linestyle='--',
    )
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].set_title('Process-Specific CPU Usage')
    axs[0].legend(loc='upper left')

    # RAM Usage
    for rank in ranks:
        df_rank = system_df[system_df['rank'] == rank]
        axs[1].plot(df_rank['timestamp'], df_rank['ram_usage'], label=f'RAM Rank {rank}')
    axs[1].plot(
        system_df['timestamp'],
        system_df.groupby('timestamp')['ram_usage'].mean(),
        label='Average RAM',
        color='black',
        linestyle='--',
    )
    axs[1].set_ylabel('RAM Usage (MB)')
    axs[1].set_title('Process-Specific RAM Usage')
    axs[1].legend(loc='upper left')

    # GPU Load
    for gpu in gpu_ids:
        df_gpu = system_df[system_df['gpu_id'] == gpu]
        axs[2].plot(df_gpu['timestamp'], df_gpu['gpu_load'], label=f'GPU {gpu} Load')
    axs[2].plot(
        system_df['timestamp'],
        system_df.groupby('timestamp')['gpu_load'].mean(),
        label='Average GPU Load',
        color='black',
        linestyle='--',
    )
    axs[2].set_ylabel('GPU Load (%)')
    axs[2].set_title('GPU Load')
    axs[2].legend(loc='upper left')

    # GPU VRAM Usage
    for gpu in gpu_ids:
        df_gpu = system_df[system_df['gpu_id'] == gpu]
        axs[3].plot(df_gpu['timestamp'], df_gpu['gpu_memory_used'], label=f'GPU {gpu} VRAM Used')
        axs[3].plot(df_gpu['timestamp'], df_gpu['gpu_memory_total'], label=f'GPU {gpu} VRAM Total')
    axs[3].plot(
        system_df['timestamp'],
        system_df.groupby('timestamp')['gpu_memory_used'].mean(),
        label='Average VRAM Used',
        color='black',
        linestyle='--',
    )
    axs[3].plot(
        system_df['timestamp'],
        system_df.groupby('timestamp')['gpu_memory_total'].mean(),
        label='Average VRAM Total',
        color='grey',
        linestyle='--',
    )
    axs[3].set_ylabel('VRAM Usage (MB)')
    axs[3].set_title('GPU VRAM Usage')
    axs[3].legend(loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
