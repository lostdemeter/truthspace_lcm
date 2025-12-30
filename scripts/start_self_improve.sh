#!/bin/bash
#
# Start the self-improvement daemon in the background
#
# Usage:
#   ./scripts/start_self_improve.sh          # Start daemon
#   ./scripts/start_self_improve.sh stop     # Stop daemon
#   ./scripts/start_self_improve.sh status   # Check status
#   ./scripts/start_self_improve.sh logs     # View logs
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.self_improve.pid"
LOG_FILE="$PROJECT_DIR/self_improve.log"

# Default settings
INTERVAL=${INTERVAL:-60}  # 1 minute between batches
SOURCES=${SOURCES:-grokipedia}  # Grokipedia is faster than Gutenberg
MIN_SCORE=${MIN_SCORE:-0.5}  # Lower threshold for more data
TOPICS_PER_CYCLE=${TOPICS_PER_CYCLE:-3}  # Fetch 3 topics per cycle

cd "$PROJECT_DIR"

start_daemon() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Self-improvement daemon already running (PID: $PID)"
            exit 1
        fi
    fi
    
    echo "Starting self-improvement daemon..."
    echo "  Interval: ${INTERVAL}s"
    echo "  Sources: $SOURCES"
    echo "  Min score: $MIN_SCORE"
    echo "  Topics per cycle: $TOPICS_PER_CYCLE"
    echo "  Log: $LOG_FILE"
    
    nohup python scripts/self_improve.py \
        --daemon \
        --interval "$INTERVAL" \
        --sources "$SOURCES" \
        --min-score "$MIN_SCORE" \
        --topics-per-cycle "$TOPICS_PER_CYCLE" \
        --log "$LOG_FILE" \
        >> "$LOG_FILE" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "Started with PID: $(cat "$PID_FILE")"
    echo ""
    echo "To stop: ./scripts/start_self_improve.sh stop"
    echo "To view logs: ./scripts/start_self_improve.sh logs"
}

stop_daemon() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping daemon (PID: $PID)..."
            kill "$PID"
            rm "$PID_FILE"
            echo "Stopped."
        else
            echo "Daemon not running (stale PID file)"
            rm "$PID_FILE"
        fi
    else
        echo "Daemon not running (no PID file)"
    fi
}

check_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Self-improvement daemon is RUNNING (PID: $PID)"
            
            # Show recent stats from log
            if [ -f "$LOG_FILE" ]; then
                echo ""
                echo "Recent activity:"
                tail -20 "$LOG_FILE" | grep -E "(Cycle|COMPLETE|Frames:|Concepts:)" | tail -10
            fi
        else
            echo "Daemon NOT running (stale PID file)"
        fi
    else
        echo "Daemon NOT running"
    fi
    
    # Show corpus stats
    CORPUS="$PROJECT_DIR/truthspace_lcm/corpus_self_improved.json"
    if [ -f "$CORPUS" ]; then
        echo ""
        echo "Corpus stats:"
        FRAMES=$(grep -c '"initiator"' "$CORPUS" 2>/dev/null || echo "0")
        echo "  Frames: $FRAMES"
        echo "  Size: $(du -h "$CORPUS" | cut -f1)"
        echo "  Modified: $(stat -c %y "$CORPUS" 2>/dev/null | cut -d. -f1)"
    fi
}

view_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
    fi
}

quality_check() {
    echo "Running quality check..."
    python scripts/quality_check.py --save-history
}

show_history() {
    python scripts/quality_check.py --history
}

set_directive() {
    DIRECTIVES_FILE="scripts/daemon_directives.json"
    
    if [ -z "$2" ]; then
        echo "Current directives:"
        cat "$DIRECTIVES_FILE"
        echo ""
        echo "Usage: $0 directive <key> <value>"
        echo "  Examples:"
        echo "    $0 directive priority_topics '[\"Quantum_mechanics\", \"Relativity\"]'"
        echo "    $0 directive priority_domains '[\"science\"]'"
        echo "    $0 directive min_quality 0.6"
        return
    fi
    
    KEY="$2"
    VALUE="$3"
    
    # Use Python to update JSON safely
    python3 -c "
import json
with open('$DIRECTIVES_FILE') as f:
    d = json.load(f)
d['$KEY'] = $VALUE
with open('$DIRECTIVES_FILE', 'w') as f:
    json.dump(d, f, indent=4)
print(f'Set $KEY = $VALUE')
print('Directives updated (takes effect next cycle)')
"
}

case "${1:-start}" in
    start)
        start_daemon
        ;;
    stop)
        stop_daemon
        ;;
    status)
        check_status
        ;;
    logs)
        view_logs
        ;;
    restart)
        stop_daemon
        sleep 1
        start_daemon
        ;;
    quality)
        quality_check
        ;;
    history)
        show_history
        ;;
    directive)
        set_directive "$@"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|restart|quality|history|directive}"
        exit 1
        ;;
esac
