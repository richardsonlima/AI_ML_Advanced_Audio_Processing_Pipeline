import os
import subprocess
from openunmix import predict
import soundfile as sf
import librosa
import torch
from demucs.separate import main as demucs_separate
from voicefixer import VoiceFixer

class Phase01:
    """Fase 1 - Reduzindo ruído com DEMUCS"""
    def __init__(self, model_names=None):
        if model_names is None:
            #model_names = ["htdemucs", "mdx_extra_q"]
            model_names = ["htdemucs"]
        self.model_names = model_names

    def separate_sources(self, input_file, output_base_path):
        for model_name in self.model_names:
            model_output_path = os.path.join(output_base_path, model_name)
            os.makedirs(model_output_path, exist_ok=True)
            print(f"[Phase 01] Separando fontes usando DEMUCS ({model_name}) no arquivo: {input_file}")
            demucs_args = ["-n", model_name, input_file, "-o", model_output_path]
            demucs_separate(demucs_args)
            print(f"[Phase 01] Separação concluída com {model_name}. Arquivos salvos em: {model_output_path}")

    def reduce_noise(self, input_file, output_dir):
        print(f"[Phase 01] Processando arquivo: {input_file}")
        os.makedirs(output_dir, exist_ok=True)
        self.separate_sources(input_file, output_dir)

class Phase02:
    """Fase 2 - Dividindo áudio em segmentos"""
    def __init__(self, segment_duration):
        self.segment_duration = segment_duration

    def split_audio(self, input_file, output_dir):
        print(f"[Phase 02] Dividindo o arquivo: {input_file} em segmentos de {self.segment_duration // 60} minuto(s)...")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_file):
            print(f"[Phase 02] Erro: Arquivo de entrada não encontrado: {input_file}")
            return []

        try:
            audio, sr = librosa.load(input_file, sr=44100, mono=True)
        except Exception as e:
            print(f"[Phase 02] Erro ao carregar o áudio com librosa: {e}")
            return []

        total_duration = librosa.get_duration(y=audio, sr=sr)
        segment_paths = []

        for i in range(0, int(total_duration), self.segment_duration):
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            start_sample = i * sr
            end_sample = min((i + self.segment_duration) * sr, len(audio))
            segment = audio[start_sample:end_sample]
            segment_path = os.path.join(output_dir, f"{base_name}_segment_{i // self.segment_duration + 1}.wav")
            try:
                sf.write(segment_path, segment, sr)
                print(f"[Phase 02] Segmento salvo: {segment_path}")
                segment_paths.append(segment_path)
            except Exception as e:
                print(f"[Phase 02] Erro ao salvar o segmento {i}: {e}")

        return segment_paths

class Phase03:
    """Fase 3 - Separação de vozes com OpenUnmix"""
    def __init__(self):
        pass

    def separate_voices(self, segment_path, output_dir):
        print(f"[Phase 03] Separando vozes do segmento: {segment_path}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            audio, sr = librosa.load(segment_path, sr=None)  # Preservar a taxa de amostragem original
            audio_tensor = torch.tensor(audio).unsqueeze(0)  # Converter para tensor com dimensão de batch
            estimates = predict.separate(audio_tensor, rate=sr)
        except Exception as e:
            print(f"[Phase 03] Erro ao separar vozes: {e}")
            return

        for instrument, audio_data in estimates.items():
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.squeeze().detach().numpy()  # Converter para numpy
            elif audio_data.ndim == 3:  # Verificar dimensões
                audio_data = audio_data.squeeze(0)  # Remover batch
            elif audio_data.ndim == 2:
                pass  # Nenhuma modificação necessária
            else:
                print(f"[Phase 03] Dimensão inesperada para áudio: {audio_data.shape}")
                continue

            output_file = os.path.join(output_dir, f"{os.path.basename(segment_path)}_{instrument}.wav")
            sf.write(output_file, audio_data.T, sr)  # Transpor para salvar no formato correto
            print(f"[Phase 03] Fonte '{instrument}' salva em: {output_file}")

class Phase04:
    """Fase 4 - Melhorar qualidade do áudio com VoiceFixer e excitador vocal"""
    def __init__(self, ffmpeg_binary):
        self.ffmpeg_binary = ffmpeg_binary
        self.voice_fixer = VoiceFixer()

    def enhance_audio(self, input_file, output_file):
        print(f"[Phase 04] Melhorando qualidade do áudio com VoiceFixer: {input_file}")
        try:
            self.voice_fixer.restore(input_file, output_file, cuda=False)
            print(f"[Phase 04] Qualidade melhorada salva em: {output_file}")
        except Exception as e:
            print(f"[Phase 04] Erro ao usar VoiceFixer: {e}")

    def apply_vocal_exciter(self, input_file, output_file):
        print(f"[Phase 04] Aplicando excitador vocal: {input_file}")
        command = [
            self.ffmpeg_binary,
            "-y", "-i", input_file,
            "-af", "bass=g=3:f=110:w=0.3,treble=g=5",  # Ajustes de excitador vocal
            output_file
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[Phase 04] Excitador vocal aplicado e salvo em: {output_file}")

    def process_audio(self, input_dir, output_dir):
        print(f"[Phase 04] Processando áudios na pasta: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".wav"):
                input_file = os.path.join(input_dir, file_name)
                base_name = os.path.splitext(file_name)[0]

                # Melhorar qualidade com VoiceFixer
                enhanced_file = os.path.join(output_dir, f"{base_name}_enhanced.wav")
                self.enhance_audio(input_file, enhanced_file)

                # Aplicar excitador vocal
                final_file = os.path.join(output_dir, f"{base_name}_final.wav")
                self.apply_vocal_exciter(enhanced_file, final_file)

class AudioProcessingPipeline:
    def __init__(self, segment_duration, ffmpeg_binary):
        self.phase01 = Phase01()
        self.phase02 = Phase02(segment_duration)
        self.phase03 = Phase03()
        self.phase04 = Phase04(ffmpeg_binary)

    def process_audio(self, input_file, output_base_dir):
        phase01_output_dir = os.path.join(output_base_dir, "Phase01_Reduced_Noise")
        os.makedirs(phase01_output_dir, exist_ok=True)

        phase02_output_dir = os.path.join(output_base_dir, "Phase02_Segments")
        os.makedirs(phase02_output_dir, exist_ok=True)

        phase03_output_dir = os.path.join(output_base_dir, "Phase03_Separated_Voices")
        os.makedirs(phase03_output_dir, exist_ok=True)

        phase04_output_dir = os.path.join(output_base_dir, "Phase04_Final_Enhancements")
        os.makedirs(phase04_output_dir, exist_ok=True)

        print(f"\n[Pipeline] Iniciando processamento do arquivo: {input_file}")
        base_name = os.path.basename(input_file)
        phase01_output_file = os.path.join(phase01_output_dir, base_name)

        # Phase 01: Reduce noise
        self.phase01.reduce_noise(input_file, phase01_output_dir)

        # Phase 02: Split audio
        segment_paths = self.phase02.split_audio(phase01_output_file, phase02_output_dir)
        if not segment_paths:
            print(f"[Pipeline] Erro: Falha na Phase 02 para o arquivo: {input_file}. Encerrando processamento.")
            return

        for segment_path in segment_paths:
            # Phase 03: Separate voices
            self.phase03.separate_voices(segment_path, phase03_output_dir)

        # Phase 04: Enhance separated voices
        self.phase04.process_audio(phase03_output_dir, phase04_output_dir)

if __name__ == "__main__":
    ffmpeg_binary = "ffmpeg"
    segment_duration = 60  # 1 minuto em segundos

    input_dir = "source_audio"
    input_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".wav")]

    output_base_dir = "Audio_Processing_Output"

    pipeline = AudioProcessingPipeline(segment_duration, ffmpeg_binary)

    for input_file in input_files:
        pipeline.process_audio(input_file, output_base_dir)
