import torch
import librosa
import numpy as np
import onnxruntime as ort
import torchaudio.transforms as TT
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def audio_volume_normalize(audio: np.ndarray, coeff: float = 0.2) -> np.ndarray:
    """
    Normalize the volume of an audio signal.
    Parameters:
        audio (numpy array): Input audio signal array.
        coeff (float): Target coefficient for normalization, default is 0.2.
    Returns:
        numpy array: The volume-normalized audio signal.
    """
    # Sort the absolute values of the audio signal
    temp = np.sort(np.abs(audio))

    # If the maximum value is less than 0.1, scale the array to have a maximum of 0.1
    if temp[-1] < 0.1:
        scaling_factor = max(
            temp[-1], 1e-3
        )  # Prevent division by zero with a small constant
        audio = audio / scaling_factor * 0.1

    # Filter out values less than 0.01 from temp
    temp = temp[temp > 0.01]
    L = temp.shape[0]  # Length of the filtered array

    # If there are fewer than or equal to 10 significant values, return the audio without further processing
    if L <= 10:
        return audio

    # Compute the average of the top 10% to 1% of values in temp
    volume = np.mean(temp[int(0.9 * L) : int(0.99 * L)])

    # Normalize the audio to the target coefficient level, clamping the scale factor between 0.1 and 10
    audio = audio * np.clip(coeff / volume, a_min=0.1, a_max=10)

    # Ensure the maximum absolute value in the audio does not exceed 1
    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio
    
def get_mel_transformer():
    mel_transformer = TT.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=640,
        hop_length=320,
        f_min=10,             # This corresponds to your 'mel_fmin'
        f_max=None,           # This corresponds to your 'mel_fmax'
        n_mels=128,
        power=1.0,            # Common default (power=2.0 is often used too)
        norm="slaney",        # Slaney normalization
        mel_scale="slaney",   # Slaney mel scale
    )
    return mel_transformer

class AudioEncoder:
    def __init__(self, decoder_paths="YaTharThShaRma999/pretrained_tts_tokenizers"):

        ## uses torch/onnx as seen fit, for some things torch is faster, others torch is faster
        device = 'cuda:0'
        wav2vec2_path = "facebook/wav2vec2-large-xlsr-53"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec2_path
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            wav2vec2_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        sess_options = ort.SessionOptions()
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0})
        ]
        
        self.s_encoder = ort.InferenceSession(f"{decoder_paths}/s_encoder.onnx", sess_options, providers=providers)
        self.q_encoder = ort.InferenceSession(f"{decoder_paths}/q_encoder.onnx", sess_options, providers=providers)
        self.mel_transformer = get_mel_transformer()
        self.ref_segment_length = 96000 
        
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 hidden state for semantics"""
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device, dtype=torch.bfloat16))

        ## wav2vec2 hidden features are merged from different layers to improve quality
        avg_feat = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return avg_feat.float()
        
    def get_ref_clip(self, wav):

        """pads to ref segment lenght"""

        ref_segment_length = self.ref_segment_length
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    @torch.inference_mode()   
    def encode(self, audio, encode_semantic=True, duration=8):

        """encodes audio file into speech tokens and context tokens"""
        audio, sr = librosa.load(audio, duration=duration, sr=16000)
        audio = audio_volume_normalize(audio)
        
        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()

        mel = self.mel_transformer(wav_ref).squeeze(1) 
        new_arr = np.array(mel.transpose(1, 2).cpu())

        global_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr}) 
        context_tokens = "".join([f"<|context_token_{i}|>" for i in global_tokens[0].squeeze()])
        if encode_semantic: 
            feat = self.extract_wav2vec2_features(audio)
            speech_tokens = self.q_encoder.run(["semantic_tokens"], {"features": feat.cpu().detach().numpy()})
            speech_tokens = "".join([f"<|speech_token_{i}|>" for i in speech_tokens[0][0]])
            return speech_tokens, context_tokens
        else:
            return context_tokens
