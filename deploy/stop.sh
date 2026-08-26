#!/usr/bin/env bash
# Stops ONLY this service.
#
# Matched on our own unique module path, never a broad pkill: this box also runs
# a Gemma server, a transliteration API, two multi-day training jobs and an
# older FlowTTS instance, and a careless pattern takes them all down.
#
# The pattern is built from fragments so this script's own command line cannot
# match it — a self-match is how a stop script kills the shell that ran it.
PAT_SVC="flowtts""\.service"
PAT_RUN="omnivoice-svc/start""\.sh"

pkill -f "$PAT_RUN" 2>/dev/null
pkill -f "$PAT_SVC" 2>/dev/null
sleep 2
if pgrep -f "$PAT_SVC" >/dev/null 2>&1; then
    echo "still running, sending SIGKILL"
    pkill -9 -f "$PAT_SVC" 2>/dev/null
    sleep 1
fi
pgrep -af "$PAT_SVC" || echo "omnivoice stopped"
