#!/bin/bash
# Signal the recorder to flush the current buffer as a chunk
PID_FILE="$HOME/.recordcli_pid"
[ -f "$PID_FILE" ] && kill -USR1 "$(cat "$PID_FILE")"
