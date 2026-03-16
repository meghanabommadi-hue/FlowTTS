import os
import wave
import struct
import argparse


def get_silence_segments(wav_path, silence_threshold=500, min_silence_duration=0.3):
    """
    Finds all silence segments anywhere in the audio.
    Returns list of (start_sec, end_sec, duration_sec) for each silent segment.
    """
    with wave.open(wav_path, 'r') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM supported, got {sampwidth*8}-bit")

    samples = struct.unpack(f"<{len(raw)//2}h", raw)

    if n_channels > 1:
        samples = [
            sum(samples[i:i+n_channels]) // n_channels
            for i in range(0, len(samples), n_channels)
        ]

    total_samples = len(samples)
    total_duration = total_samples / framerate
    min_silence_samples = int(min_silence_duration * framerate)

    # Find contiguous silence runs
    segments = []
    in_silence = False
    silence_start = 0

    for i, s in enumerate(samples):
        if abs(s) <= silence_threshold:
            if not in_silence:
                in_silence = True
                silence_start = i
        else:
            if in_silence:
                length = i - silence_start
                if length >= min_silence_samples:
                    segments.append((
                        silence_start / framerate,
                        i / framerate,
                        length / framerate,
                    ))
                in_silence = False

    # Handle silence that runs to the end
    if in_silence:
        length = total_samples - silence_start
        if length >= min_silence_samples:
            segments.append((
                silence_start / framerate,
                total_duration,
                length / framerate,
            ))

    return total_duration, segments


def scan_folder(audio_dir, silence_threshold=500, min_silence_duration=0.3,
                min_total_silence=0.5, ext=".wav"):
    results = []
    files = [f for f in os.listdir(audio_dir) if f.endswith(ext)]
    print(f"Scanning {len(files)} files in {audio_dir} ...")

    for fname in files:
        fpath = os.path.join(audio_dir, fname)
        try:
            total, segs = get_silence_segments(fpath, silence_threshold, min_silence_duration)
            total_silence = sum(d for _, _, d in segs)
            if total_silence >= min_total_silence:
                results.append((fname, total, total_silence, segs))
        except Exception as e:
            print(f"  ERROR {fname}: {e}")

    results.sort(key=lambda x: -x[2])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find WAV files with silence anywhere in the audio.")
    parser.add_argument("audio_dir", help="Folder containing .wav files")
    parser.add_argument("--threshold", type=int, default=500,
                        help="Amplitude threshold for silence (0-32767, default 500)")
    parser.add_argument("--min-segment", type=float, default=0.3,
                        help="Minimum duration (sec) to count a run as silence (default 0.3)")
    parser.add_argument("--min-total", type=float, default=0.5,
                        help="Minimum total silence (sec) to flag a file (default 0.5)")
    parser.add_argument("--show-segments", action="store_true",
                        help="Print each silence segment's timestamps")
    parser.add_argument("--output", type=str, default=None,
                        help="Save flagged file list to this path")
    args = parser.parse_args()

    flagged = scan_folder(args.audio_dir, args.threshold, args.min_segment, args.min_total)

    print(f"\nFound {len(flagged)} files with >={args.min_total}s total silence:\n")
    print(f"{'filename':<70} {'total':>8} {'silence':>9} {'segments':>9}")
    print("-" * 100)
    for fname, total, total_silence, segs in flagged:
        print(f"{fname:<70} {total:>7.2f}s {total_silence:>8.2f}s {len(segs):>8}x")
        if args.show_segments:
            for start, end, dur in segs:
                location = "trailing" if end >= total - 0.05 else "leading" if start < 0.05 else "internal"
                print(f"    [{location}]  {start:.2f}s → {end:.2f}s  ({dur:.2f}s)")

    if args.output:
        with open(args.output, "w") as f:
            for fname, total, total_silence, segs in flagged:
                f.write(f"{fname}\t{total:.2f}\t{total_silence:.2f}\t{len(segs)}\n")
        print(f"\nSaved to {args.output}")

    print(f"\nTotal flagged: {len(flagged)}")
