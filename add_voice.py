#!/usr/bin/env python3
"""DEPRECATED — voices are now precomputed npz artifacts, not config edits.

The old MiraTTS voice workflow (editing VOICE_REF_AUDIO / _VOICE_CACHE_MAP) is
gone. OmniVoice voices are reusable `voices/<alias>.npz` files built once, offline.

Use the voice-clone CLI instead:

    # add one voice (ref_text auto-transcribed if omitted)
    python -m flowtts.voices.clone --add <alias> --ref-audio sample_files/<file>.wav

    # build every voice from sample_files/ (+ voices/manifest.json)
    python -m flowtts.voices.clone --build-all

    # list installed voices
    python -m flowtts.voices.clone --list

A request selects a voice by alias via the WebSocket `voice_id` field.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(2)
