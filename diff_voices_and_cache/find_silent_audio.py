import os
import wave
import struct
import argparse

def get_trailing_silence(wav_path, silence_threshold=500, sample_rate=None):
    """
    Returns (total_duration, trailing_silence_duration, speech_duration) in seconds.
    silence_threshold: amplitude below this is considered silence (0-32767 for 16-bit PCM).
    """
    with wave.open(wav_path, 'r') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM supported, got {sampwidth*8}-bit: {wav_path}")

    samples = struct.unpack(f"<{len(raw)//2}h", raw)

    # Mono: average channels
    if n_channels > 1:
        samples = [
            sum(samples[i:i+n_channels]) // n_channels
            for i in range(0, len(samples), n_channels)
        ]

    total_samples = len(samples)
    total_duration = total_samples / framerate

    # Find last non-silent sample
    last_speech = 0
    for i in range(total_samples - 1, -1, -1):
        if abs(samples[i]) > silence_threshold:
            last_speech = i
            break

    speech_duration = (last_speech + 1) / framerate
    trailing_silence = total_duration - speech_duration

    return total_duration, trailing_silence, speech_duration


def find_silent_files(audio_dir, silence_threshold=500, min_trailing_silence=2.0, ext=".wav"):
    results = []

    files = [f for f in os.listdir(audio_dir) if f.endswith(ext)]
    print(f"Scanning {len(files)} files in {audio_dir} ...")

    for fname in files:
        fpath = os.path.join(audio_dir, fname)
        try:
            total, trailing, speech = get_trailing_silence(fpath, silence_threshold)
            if trailing >= min_trailing_silence:
                results.append((fname, total, trailing, speech))
        except Exception as e:
            print(f"  ERROR {fname}: {e}")

    results.sort(key=lambda x: -x[2])  # sort by trailing silence descending
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find WAV files with excessive trailing silence.")
    parser.add_argument("audio_dir", help="Folder containing .wav files")
    parser.add_argument("--threshold", type=int, default=500,
                        help="Amplitude threshold for silence (0-32767, default 500)")
    parser.add_argument("--min-silence", type=float, default=2.0,
                        help="Minimum trailing silence in seconds to flag (default 2.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save flagged file list to this path")
    args = parser.parse_args()

    flagged = find_silent_files(args.audio_dir, args.threshold, args.min_silence)

    print(f"\nFound {len(flagged)} files with >{args.min_silence}s trailing silence:\n")
    print(f"{'filename':<70} {'total':>8} {'speech':>8} {'silence':>8}")
    print("-" * 100)
    for fname, total, trailing, speech in flagged:
        print(f"{fname:<70} {total:>7.2f}s {speech:>7.2f}s {trailing:>7.2f}s")

    if args.output:
        with open(args.output, "w") as f:
            for fname, total, trailing, speech in flagged:
                f.write(f"{fname}\t{total:.2f}\t{speech:.2f}\t{trailing:.2f}\n")
        print(f"\nSaved to {args.output}")

    print(f"\nTotal flagged: {len(flagged)}")
