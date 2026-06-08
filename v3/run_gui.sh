#!/bin/bash
# Helper runner script to launch the Expert Dataset Review GUI
# Binding only to localhost (127.0.0.1) as required by security guidelines
echo "Starting Expert Dataset Review GUI on local environment..."
python3 "$(dirname "$0")/expert_gui/app.py"
