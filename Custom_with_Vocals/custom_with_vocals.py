import os
from script_debug import Phase02, Phase03, Phase04

# Configurações do script
INPUT_FILE = "2h30m3h08m-vocals.wav"
SEGMENT_DURATION = 60  # Duração dos segmentos em segundos
FFMPEG_BINARY = "ffmpeg"

# Diretórios de saída
BASE_OUTPUT_DIR = "Custom_Processing_Output"
PHASE02_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Phase02_Segments")
PHASE03_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Phase03_Separated_Voices")
PHASE04_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "Phase04_Final_Enhancements")

# Certifique-se de que os diretórios de saída existam
os.makedirs(PHASE02_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHASE03_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHASE04_OUTPUT_DIR, exist_ok=True)

# Inicializando as fases
phase02 = Phase02(SEGMENT_DURATION)
phase03 = Phase03()
phase04 = Phase04(FFMPEG_BINARY)

# Fase 02: Dividindo o áudio
print(f"[Custom Script] Iniciando Phase02: Divisão do áudio {INPUT_FILE}")
segment_paths = phase02.split_audio(INPUT_FILE, PHASE02_OUTPUT_DIR)

if not segment_paths:
    print("[Custom Script] Erro: Não foi possível dividir o áudio. Encerrando o processo.")
    exit(1)

# Fase 03: Separando vozes
print("[Custom Script] Iniciando Phase03: Separação de vozes")
for segment_path in segment_paths:
    phase03.separate_voices(segment_path, PHASE03_OUTPUT_DIR)

# Fase 04: Melhorando qualidade do áudio
print("[Custom Script] Iniciando Phase04: Melhoria da qualidade do áudio")
phase04.process_audio(PHASE03_OUTPUT_DIR, PHASE04_OUTPUT_DIR)

print("[Custom Script] Processamento concluído. Arquivos finais disponíveis em:", PHASE04_OUTPUT_DIR)
