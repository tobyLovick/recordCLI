#!/bin/bash
CURRENT="$HOME/recordCLI/notes/current.txt"
OFFSET_FILE="$HOME/.recordcli_offset"
TABLET_FILE="/tmp/tablet_latex.md"
TABLET_OFFSET_FILE="$HOME/.tablet_latex_offset"

# ── collect new transcript content ───────────────────────────────
transcript_content=""
if [ -f "$CURRENT" ]; then
    current_size=$(wc -c < "$CURRENT")
    offset=0
    [ -f "$OFFSET_FILE" ] && offset=$(cat "$OFFSET_FILE")
    if [ "$current_size" -gt "$offset" ]; then
        transcript_content=$(tail -c +"$((offset + 1))" "$CURRENT")
        echo "$current_size" > "$OFFSET_FILE"
    fi
fi

# ── collect new tablet LaTeX (offset-tracked, like transcript) ────
tablet_content=""
if [ -f "$TABLET_FILE" ]; then
    current_size=$(wc -c < "$TABLET_FILE")
    offset=0
    [ -f "$TABLET_OFFSET_FILE" ] && offset=$(cat "$TABLET_OFFSET_FILE")
    if [ "$current_size" -gt "$offset" ]; then
        tablet_content=$(tail -c +"$((offset + 1))" "$TABLET_FILE")
        echo "$current_size" > "$TABLET_OFFSET_FILE"
    fi
fi

# ── exit if nothing new ───────────────────────────────────────────
[ -z "$transcript_content" ] && [ -z "$tablet_content" ] && exit 0

# ── build additionalContext ───────────────────────────────────────
python3 -c "
import json, sys

transcript = sys.argv[1]
tablet = sys.argv[2]

parts = []
if transcript:
    parts.append('[Live transcript — act on any requests or questions from the researcher, ignore background conversation]:\n' + transcript)
if tablet:
    parts.append('[Handwritten math from tablet — the researcher just drew and processed this expression]:\n' + tablet)

context = '\n\n'.join(parts)
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'UserPromptSubmit',
        'additionalContext': context
    }
}))
" "$transcript_content" "$tablet_content"
