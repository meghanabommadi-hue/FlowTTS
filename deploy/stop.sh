#!/usr/bin/env bash
# Stops ONLY this service: the supervisor first, then its child.
#
# The supervisor is killed FIRST, and by PID from a pidfile. Killing the child
# first just hands the restart loop something to restart, and matching the
# supervisor by command line is unreliable — it runs as `./start.sh`, so a
# pattern built from the absolute path never matches, while a looser pattern
# catches things it must not.
#
# Two self-match hazards this script has to avoid, both of which have actually
# bitten here:
#
#   * the service pattern is assembled from fragments, so this script's own
#     command line cannot contain it. A `pkill -f` that matches the shell
#     running the pkill kills that shell.
#   * the fallback scan below skips this process and every one of its
#     ancestors, and requires a candidate to *be* `bash …start.sh` (exactly two
#     arguments) rather than merely mention it. Without both checks it matches
#     an operator's own interactive shell whenever they happen to be sitting in
#     this directory, or the `bash -c` of an ssh command that mentions the
#     script — and kills it.
set -u
DIR=/root/omnivoice-svc
PIDFILE=$DIR/omnivoice.pid
CHILDFILE=$DIR/omnivoice.child.pid
MOD="flowtts""\.service"

# ---- 1. the supervisor, so nothing restarts behind us ----------------------
if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "${pid:-}" ]; then
        kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PIDFILE"
fi

# Ancestors of this shell, which must never be killed.
protected=" $$ "
walk=$$
while [ "$walk" -gt 1 ]; do
    walk=$(ps -o ppid= -p "$walk" 2>/dev/null | tr -d ' ')
    [ -n "${walk:-}" ] || break
    protected="$protected$walk "
done

# Supervisors predating the pidfile: identified by working directory, which no
# other service on this box shares.
for pid in $(pgrep -x bash 2>/dev/null); do
    case "$protected" in *" $pid "*) continue ;; esac
    [ -r "/proc/$pid/cwd" ] || continue
    [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$DIR" ] || continue
    # Must BE the launcher: `bash ./start.sh` is two arguments. `bash -c …` is
    # not, however much it mentions start.sh.
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    set -- $cmd
    [ "$#" -eq 2 ] || continue
    case "$2" in *start.sh) ;; *) continue ;; esac
    kill -9 "$pid" 2>/dev/null
done

# ---- 2. the service itself -------------------------------------------------
if [ -f "$CHILDFILE" ]; then
    pid=$(cat "$CHILDFILE" 2>/dev/null)
    if [ -n "${pid:-}" ]; then
        kill "$pid" 2>/dev/null
    fi
    rm -f "$CHILDFILE"
fi
pkill -f "$MOD" 2>/dev/null
sleep 3
if pgrep -f "$MOD" >/dev/null 2>&1; then
    pkill -9 -f "$MOD" 2>/dev/null
    sleep 2
fi

# Test on pgrep's exit status, not its count: `pgrep -c` prints 0 AND exits
# non-zero when it finds nothing, so `$(pgrep -c … || echo 0)` yields "0\n0".
if pgrep -f "$MOD" >/dev/null 2>&1; then
    echo "WARNING: still running"
    pgrep -af "$MOD"
    exit 1
fi
echo "omnivoice stopped"
