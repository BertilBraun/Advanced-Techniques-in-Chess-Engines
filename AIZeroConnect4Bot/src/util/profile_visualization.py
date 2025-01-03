import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_usage(filename='usage.csv'):
    with open(filename, 'r') as file:
        lines = file.readlines()

    split_index = None
    for i, line in enumerate(lines):
        if line.startswith('timestamp'):
            split_index = i
            break

    if split_index is None:
        raise ValueError('No system usage section found.')

    events_lines = lines[:split_index]
    system_lines = lines[split_index:]

    with open('events.csv', 'w') as file:
        file.writelines(events_lines)

    with open('system_usage.csv', 'w') as file:
        file.writelines(system_lines)

    events_df = pd.read_csv('events.csv', header=None, names=['timestamp', 'event_type', 'event_name'])
    system_df = pd.read_csv('system_usage.csv', parse_dates=['timestamp'])

    return events_df, system_df


def parse_events(df):
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


def visualize(usage_file='usage.csv'):
    events_df, system_df = load_usage(usage_file)
    events = parse_events(events_df)
    system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])

    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    axs[0].plot(system_df['timestamp'], system_df['cpu_percent'], label='CPU Usage (%)', color='tab:blue')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].set_title('Process-Specific CPU Usage')
    axs[0].legend(loc='upper left')

    axs[1].plot(system_df['timestamp'], system_df['ram_usage'], label='RAM Usage (MB)', color='tab:green')
    axs[1].set_ylabel('RAM Usage (MB)')
    axs[1].set_title('Process-Specific RAM Usage')
    axs[1].legend(loc='upper left')

    axs[2].plot(system_df['timestamp'], system_df['gpu_load'], label='GPU Load (%)', color='tab:red')
    axs[2].set_ylabel('GPU Load (%)')
    axs[2].set_title('GPU Load')
    axs[2].legend(loc='upper left')

    axs[3].plot(system_df['timestamp'], system_df['gpu_memory_used'], label='VRAM Used (MB)', color='tab:purple')
    axs[3].plot(system_df['timestamp'], system_df['gpu_memory_total'], label='VRAM Total (MB)', color='tab:orange')
    axs[3].set_ylabel('VRAM Usage (MB)')
    axs[3].set_title('GPU VRAM Usage')
    axs[3].legend(loc='upper left')

    event_identifiers = {}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']

    for event in events:
        name = event['event_name']
        if name not in event_identifiers:
            event_identifiers[name] = (colors.pop(0), name.replace('_', ' ').title())

    patches = []
    labels = []

    for event in events:
        name = event['event_name']
        start = event['start']
        end = event['end']
        color, label = event_identifiers[name]

        for ax in axs:
            ax.axvspan(start, end, color=color, alpha=0.3)

        if label not in labels:
            patches.append(mpatches.Patch(color=color, alpha=0.3, label=label))
            labels.append(label)

    handles, existing_labels = axs[-1].get_legend_handles_labels()
    handles.extend(patches)
    existing_labels.extend(labels)

    axs[-1].legend(handles=handles, loc='upper left')
    axs[-1].set_xlabel('Time')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize()
