#!/usr/bin/env python3
"""
Generate target sentences using VoxCPM2 voice cloning.
Each voice gets every sentence → <voice>_<idx:02d>.wav
Output: ~/FlowTTS/test/voiced_output/
"""
import asyncio
import io
import struct
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/flow_voxcpm")

import numpy as np
from nanovllm_voxcpm import VoxCPM

VOICES = {
    "anika_vb": {
        "file": "sample_files/anika_vb.mp3",
        "ref_text": "janta ki sarkar janta dwara janta keliye prithvi se nahi mitegi",
    },
    "anika2_vb": {
        "file": "sample_files/anika2_vb.mp3",
        "ref_text": "namasthe, call karne keliye aapka dhanyavaad, chinta mat kijiye, mein abhi aapki madad karti hun",
    },
    "gargi_vb": {
        "file": "sample_files/gargi_vb.mp3",
        "ref_text": "samluchu sir, uh order delivery mein delay dikh raha hain, mein courier se abhi confirm kar rahi hun, app please hold kijiye",
    },
    "monika_vb": {
        "file": "sample_files/monika_vb.mp3",
        "ref_text": "hello sir, mein monika bol rhi hun customer support se, batayiye kya dikkat aa rhi hain, mein poori koshish karungi aapki help karne ki",
    },
    "saavi_vb": {
        "file": "sample_files/saavi_vb.mp3",
        "ref_text": "hello sir, i hope sab theek chal raha hoga, batayiye mein aapki kis tarah se madad kar sakti hun",
    },
    "tara_firm": {
        "file": "sample_files/tara_firm.mp3",
        "ref_text": "I am not here to argue, but when the same account stays silent for two months, no payment no call no email the recovery team gets activated",
    },
    "zara_vb": {
        "file": "sample_files/zara_vb.mp3",
        "ref_text": "theek hai, bas pehele ek choti si confirmation chahiye. haan bas wahi detail, ab aap ka request smoothly aage badh jaayega",
    },
}

SENTENCES = [
    "नमस्ते, मैं अग्रिम से साक्षी बात कर रही हूँ | क्या आपकी बीज, कृषि उपकरण और खाद दवाई की दुकान है?",
    "क्या हम अभी भी जुड़े हुए हैं जी?",
    "लगता है अभी आप busy हैं, मैं आपको बाद में फ़ोन करती हूँ। धन्यवाद, आपका दिन शुभ हो! Goodbye",
    "ठीक है, धन्यवाद आपका दिन शुभ हो. Goodbye.",
    "सर, local market से जो खाद, बीज और दवाई लेते होंगे — उस पर आठ हज़ार रुपये तक का discount मिलेगा, और वही सामान local market से आठ टक्का सस्ता भी पड़ेगा। जानना चाहोगे कैसे?",
    "सर, अग्रिम भारत का सबसे बड़ा खाद, बीज और दवाई का online supplier app है — N A C L, Crystal, Atul, H P M, Biostadt सब यहाँ मिलते हैं।",
    "Local market से जो भी सामान लेते हो — वही यहाँ छह से आठ टक्का सस्ता मिलेगा, सीधे दुकान पर delivery। पहले पाँच orders पर आठ हज़ार रुपये तक का discount — बचत ही बचत।",
    "एक लाख retailers रोज़ हमसे इसीलिए order कर रहे हैं। बस तीस seconds दीजिए — app download करवा देती हूँ, आपके number पर offer active हो जाएगा।",
]

OUT_DIR = Path.home() / "FlowTTS/test/voiced_output"


def pcm_to_wav(pcm: np.ndarray, sr: int) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm16)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm16)))
    buf.write(pcm16)
    return buf.getvalue()


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VoxCPM2 model...")
    server = VoxCPM.from_pretrained(
        "/home/ubuntu/voxcpm/model",
        inference_timesteps=6,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        devices=[0],
    )
    await server.wait_for_ready()
    sr = int(dict(await server.get_model_info())["sample_rate"])
    print(f"Model ready. sample_rate={sr}Hz\n")

    # Encode all voices
    print("Encoding reference voices...")
    encoded: dict[str, tuple[bytes, str]] = {}
    for name, info in VOICES.items():
        path = Path.home() / "FlowTTS" / info["file"]
        raw = path.read_bytes()
        latents = await server.encode_latents(raw, "mp3")
        ref_text = info["ref_text"]
        n_frames = len(latents) // 4 // 64
        print(f"  [{name}] {n_frames} frames  ref={repr(ref_text[:60])}")
        encoded[name] = (latents, ref_text)
    print()

    # Generate: each voice × each sentence
    total = len(VOICES) * len(SENTENCES)
    saved = 0
    for v_idx, (voice_name, (latents, ref_text)) in enumerate(encoded.items()):
        print(f"── {voice_name} ──")
        voice_dir = OUT_DIR / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        for s_idx, sentence in enumerate(SENTENCES):
            out_path = voice_dir / f"{voice_name}_{s_idx:02d}.wav"

            chunks = []
            async for chunk in server.generate(
                target_text=sentence,
                prompt_latents=latents,
                prompt_text=ref_text,
                cfg_value=2.0,
                temperature=1.0,
            ):
                if isinstance(chunk, dict):
                    continue
                chunks.append(np.asarray(chunk, dtype=np.float32))

            if chunks:
                pcm = np.concatenate(chunks)
                maxamp = int(np.max(np.abs(pcm)) * 32767)
                dur = len(pcm) / sr
                out_path.write_bytes(pcm_to_wav(pcm, sr))
                status = "OK" if maxamp > 100 else "SILENT"
                print(f"  [{s_idx:02d}] {dur:.2f}s  amp={maxamp:>6}  [{status}]  {out_path.name}")
                saved += 1
            else:
                print(f"  [{s_idx:02d}] NO AUDIO  {out_path.name}")
        print()

    await server.stop()
    print(f"Done. {saved}/{total} files → {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
