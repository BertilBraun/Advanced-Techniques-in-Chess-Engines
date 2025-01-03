import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_events(filename='events.csv'):
    """
    Parses the events log and returns a list of events with start and end times.
    """
    df = pd.read_csv(filename, header=None, names=['timestamp', 'event_type', 'event_name'], skiprows=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    events = []
    ongoing_events = {}

    for _, row in df.iterrows():
        if row['event_type'] == 'START':
            ongoing_events[row['event_name']] = row['timestamp']
        elif row['event_type'] == 'END':
            start_time = ongoing_events.pop(row['event_name'], None)
            if start_time:
                events.append({'event_name': row['event_name'], 'start': start_time, 'end': row['timestamp']})

    return events


def visualize(system_usage_file='system_usage.csv', events_file='events.csv'):
    # Read system usage data
    system_df = pd.read_csv(system_usage_file, parse_dates=['timestamp'])

    # Read events
    events = parse_events(events_file)

    # Convert timestamp to datetime
    system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])

    # Set up the plot with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    # Plot CPU Usage
    axs[0].plot(system_df['timestamp'], system_df['cpu_percent'], label='CPU Usage (%)', color='tab:blue')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].set_title('Process-Specific CPU Usage')
    axs[0].legend(loc='upper left')

    # Plot RAM Usage
    axs[1].plot(system_df['timestamp'], system_df['ram_usage'], label='RAM Usage (MB)', color='tab:green')
    axs[1].set_ylabel('RAM Usage (MB)')
    axs[1].set_title('Process-Specific RAM Usage')
    axs[1].legend(loc='upper left')

    # Plot GPU Load
    axs[2].plot(system_df['timestamp'], system_df['gpu_load'], label='GPU Load (%)', color='tab:red')
    axs[2].set_ylabel('GPU Load (%)')
    axs[2].set_title('GPU Load')
    axs[2].legend(loc='upper left')

    # Plot VRAM Usage
    axs[3].plot(system_df['timestamp'], system_df['gpu_memory_used'], label='VRAM Used (MB)', color='tab:purple')
    axs[3].plot(system_df['timestamp'], system_df['gpu_memory_total'], label='VRAM Total (MB)', color='tab:orange')
    axs[3].set_ylabel('VRAM Usage (MB)')
    axs[3].set_title('GPU VRAM Usage')
    axs[3].legend(loc='upper left')

    # Add background color for events
    for event in events:
        event_name = event['event_name']
        start = event['start']
        end = event['end']

        if event_name == 'self_play':
            color = 'green'
        elif event_name == 'training':
            color = 'orange'
        elif event_name == 'dataset_loading':
            color = 'blue'
        else:
            color = 'grey'

        for ax in axs:
            ax.axvspan(start, end, color=color, alpha=0.3)

    # Create custom legends for background colors
    self_play_patch = mpatches.Patch(color='green', alpha=0.3, label='Self-Play')
    training_patch = mpatches.Patch(color='orange', alpha=0.3, label='Training')
    dataset_loading_patch = mpatches.Patch(color='blue', alpha=0.3, label='Dataset Loading')

    # Combine existing legends with event legends
    handles, labels = axs[-1].get_legend_handles_labels()
    handles.extend([self_play_patch, training_patch, dataset_loading_patch])
    labels.extend(['Self-Play', 'Training', 'Dataset Loading'])

    # Add the combined legend to the last subplot
    axs[-1].legend(handles=handles, loc='upper left')

    # Set common x-label
    axs[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
