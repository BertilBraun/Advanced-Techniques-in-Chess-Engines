#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define the session name
SESSION="chess_tmux_session"

# Function to display error messages
error() {
    echo "Error: $1" >&2
    exit 1
}

# Function to print the current tmux state
print_tmux_state() {
    echo "---- Tmux Session State ----"
    tmux list-windows -t "$SESSION" -F 'Window #I: #W'
    tmux list-panes -t "${SESSION}:main" -F 'Pane #I: #{pane_id}'
    tmux list-panes -t "${SESSION}:tensorboard" -F 'Pane #I: #{pane_id}'
    echo "----------------------------"
}

# Check if the session already exists
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Creating new tmux session: $SESSION"

    # Create a new session in detached mode with the first window named 'main'
    tmux new-session -d -s "$SESSION" -n main || error "Failed to create new session."

    echo "Created session '$SESSION' with window 'main'."

    # Split the 'main' window vertically with a 70/30 ratio
    tmux split-window -h -p 30 -t "${SESSION}:main" || error "Failed to split window."

    echo "Split window 'main' into two panes (70% and 30%)."

    # Optional: Wait a moment to ensure tmux processes the split
    sleep 0.5

    # In the second pane (pane1), run "qi"
    tmux send-keys -t "${SESSION}:main.1" "qi" C-m || error "Failed to send 'qi' to pane1."

    echo "Sent 'qi' command to pane1 of window 'main'."

    # Optional: Wait a moment to ensure the command is sent
    sleep 0.5

    # Create a new window named 'tensorboard'
    tmux new-window -t "$SESSION" -n tensorboard || error "Failed to create window 'tensorboard'."

    echo "Created window 'tensorboard'."

    # Optional: Wait a moment to ensure the window is created
    sleep 0.5

    # In the 'tensorboard' window, run "tb logs"
    tmux send-keys -t "${SESSION}:tensorboard.0" "tb logs" C-m || error "Failed to send 'tb logs' to 'tensorboard' window."

    echo "Sent 'tb logs' command to window 'tensorboard'."

    # Explicitly select the 'main' window and pane0 to set focus
    tmux select-window -t "${SESSION}:main" || error "Failed to select window 'main'."
    tmux select-pane -t "${SESSION}:main.0" || error "Failed to select pane 'main.0'."

    echo "Set focus to pane0 of window 'main'."

    # Print the current state of the tmux session for debugging
    print_tmux_state

    echo "Tmux session '$SESSION' setup complete."
else
    echo "Tmux session '$SESSION' already exists. Attaching to the session."

    # Optional: Print the current state of the tmux session for debugging
    print_tmux_state
fi

# Attach to the session
tmux attach -t "$SESSION" || error "Failed to attach to session '$SESSION'."
