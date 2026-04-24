#!/bin/bash
CURRENT="$HOME/recordCLI/notes/current.txt"
OFFSET_FILE="$HOME/.recordcli_offset"

[ -f "$CURRENT" ] || exit 0

current_size=$(wc -c < "$CURRENT")
offset=0
[ -f "$OFFSET_FILE" ] && offset=$(cat "$OFFSET_FILE")

[ "$current_size" -le "$offset" ] && exit 0

new_content=$(tail -c +"$((offset + 1))" "$CURRENT")
echo "$current_size" > "$OFFSET_FILE"

[ -z "$new_content" ] && exit 0

python3 -c "
import json, sys
content = sys.stdin.read()
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'UserPromptSubmit',
        'additionalContext': '[New voice transcript since last message]:\n' + content
    }
}))" <<< "$new_content"
