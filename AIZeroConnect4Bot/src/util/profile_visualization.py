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

    # Plot CPU and GPU usage
    fig, ax1 = plt.subplots(figsize=(15, 8))

    ax1.set_title('System Usage During Self-Play and Training')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Usage (%)', color='tab:blue')
    ax1.plot(system_df['timestamp'], system_df['cpu_percent'], label='CPU Usage (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('GPU Load (%)', color='tab:red')
    ax2.plot(system_df['timestamp'], system_df['gpu_load'], label='GPU Load (%)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add background color for events
    for event in events:
        if event['event_name'] == 'self_play':
            color = 'green'
        elif event['event_name'] == 'training':
            color = 'orange'
        else:
            color = 'grey'

        ax1.axvspan(event['start'], event['end'], color=color, alpha=0.3)

    # Create custom legends for background colors
    self_play_patch = mpatches.Patch(color='green', alpha=0.3, label='Self-Play')
    training_patch = mpatches.Patch(color='orange', alpha=0.3, label='Training')
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([self_play_patch, training_patch])
    ax1.legend(handles=handles, loc='upper left')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
