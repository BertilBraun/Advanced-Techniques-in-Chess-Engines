#!/bin/bash

# Define the session name
SESSION="chess_tmux_session"

# Check if the session already exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    # Create a new session in detached mode with main
    tmux new-session -d -s $SESSION -n main

    # Create window2 and run "tb logs"
    tmux new-window -t $SESSION:2 -n tensorboard "tb logs"

    # Split main vertically with a 70/30 ratio
    tmux split-window -h -p 30 -t ${SESSION}:main

    # In the second pane (30%), run the "qi" program
    tmux send-keys -t ${SESSION}:main.2 "qi" C-m

    # Ensure the first pane is selected
    tmux select-pane -t ${SESSION}:main.1
fi

# Attach to the session
tmux attach -t $SESSION
