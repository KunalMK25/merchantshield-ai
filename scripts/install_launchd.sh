#!/bin/bash
# Schedule the daily retraining on macOS with a launchd agent (cron cannot read ~/Desktop, ~/Documents or
# ~/Downloads without Full Disk Access, and fails silently).
#
#   scripts/install_launchd.sh            install and load the agent
#   scripts/install_launchd.sh --remove   unload and delete it
#   scripts/install_launchd.sh --print    show the plist that would be installed, change nothing
#
# macOS may ask once whether python may access the folder the project is in. Keeping the project outside
# Desktop, Documents and Downloads avoids that prompt.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.merchantshield.retrain"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$PROJECT/ml/data_store"
DOMAIN="gui/$(id -u)"

render() { sed -e "s|__PROJECT__|$PROJECT|g" -e "s|__LOGDIR__|$LOGDIR|g" "$PROJECT/scripts/launchd/$LABEL.plist.template"; }

case "${1:-install}" in
  --print) render ;;
  --remove)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $LABEL" ;;
  install)
    [ -x "$PROJECT/.venv/bin/python" ] || { echo "No virtualenv at $PROJECT/.venv. Create it first (see README)."; exit 1; }
    mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"
    render > "$PLIST"
    plutil -lint "$PLIST" >/dev/null
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Installed $LABEL: runs daily at 02:00, logs to $LOGDIR/retrain.log"
    echo "Run it once now to check:   launchctl kickstart -k $DOMAIN/$LABEL"
    echo "Remove it later with:       $0 --remove" ;;
  *) echo "usage: $0 [--print|--remove]"; exit 2 ;;
esac
