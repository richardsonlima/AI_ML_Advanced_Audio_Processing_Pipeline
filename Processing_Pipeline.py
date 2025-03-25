#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from openunmix import predict
import soundfile as sf
import librosa
import torch
from demucs.separate import main as demucs_separate
from voicefixer import VoiceFixer

# ANSI color codes for terminal formatting
RESET   = "\033[0m"
BLUE    = "\033[94m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

def log_info(message):
    print(f"{GREEN}==> {message}{RESET}")

def log_warning(message):
    print(f"{YELLOW}==> {message}{RESET}")

def log_error(message):
    print(f"{RED}==> {message}{RESET}")

def log_phase(phase, message):
    print(f"{CYAN}==> [{phase}] {message}{RESET}")

def show_help_menu():
    print(f"""
{CYAN}==================================================
       AUDIO PROCESSING PIPELINE - HELP MENU
=================================================={RESET}

{GREEN}DESCRIPTION:{RESET}
This script performs audio processing in 4 phases:
1. Noise reduction using DEMUCS
2. Audio segmentation
3. Voice separation with OpenUnmix
4. Quality enhancement with VoiceFixer and vocal exciter

{YELLOW}SYSTEM REQUIREMENTS:{RESET}
- Python 3.8 or higher
- PIP packages: torch, librosa, soundfile, openunmix, demucs, voicefixer
- FFmpeg installed and available in PATH

{MAGENTA}DIRECTORY STRUCTURE:{RESET}
- Input directory: {BLUE}./source_audio/{RESET}
  - Should contain .wav files for processing
- Output directory: {BLUE}./Audio_Processing_Output/{RESET}
  - Automatically created structure:
    - Phase01_Reduced_Noise/
    - Phase02_Segments/
    - Phase03_Separated_Voices/
    - Phase04_Final_Enhancements/

{GREEN}FILE FORMATS:{RESET}
- Input: .wav files (any sample rate)
- Output: multiple .wav files organized by phase

{YELLOW}CONFIGURABLE PARAMETERS:{RESET}
- Segment duration: 60 seconds (modifiable in code)
- DEMUCS models used: htdemucs and mdx_extra_q

{MAGENTA}INSTALLATION INSTRUCTIONS:{RESET}
1. First install system dependencies:
   {GREEN}sudo apt-get install ffmpeg{RESET} (Linux)
   or {GREEN}brew install ffmpeg{RESET} (Mac)
   or download from {BLUE}https://ffmpeg.org/{RESET} (Windows/Mac)

2. Then install Python packages:
   {GREEN}pip install torch librosa soundfile openunmix demucs voicefixer{RESET}

3. For GPU acceleration (optional):
   {GREEN}pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu117{RESET}

{CYAN}USAGE INSTRUCTIONS:{RESET}
1. Place your .wav files in {BLUE}./source_audio/{RESET}
2. Run the script:
   {GREEN}python Processing_Pipeline.py{RESET}
3. Processing will be automatic:
   - Each phase will show colored logs in terminal
   - Intermediate files will be saved in subfolders

{RED}TROUBLESHOOTING:{RESET}
- Verify all requirements are installed
- Check write permissions in directories
- Input files must be valid .wav files
- For VoiceFixer errors, try running without CUDA

{GREEN}EXAMPLE USAGE:{RESET}
1. Create {BLUE}source_audio{RESET} folder and add audio.wav
2. Run: {GREEN}python Processing_Pipeline.py{RESET}
3. Results will be in {BLUE}Audio_Processing_Output/{RESET}

{CYAN}=================================================={RESET}
""")

def log_info(message):
    print(f"{GREEN}==> {message}{RESET}")

def log_warning(message):
    print(f"{YELLOW}==> {message}{RESET}")

def log_error(message):
    print(f"{RED}==> {message}{RESET}")

def log_phase(phase, message):
    print(f"{CYAN}==> [{phase}] {message}{RESET}")

def check_ffmpeg():
    """Check if FFmpeg is available in the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "ffmpeg"
    except FileNotFoundError:
        possible_paths = [
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "C:\\FFmpeg\\bin\\ffmpeg.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

def convert_to_wav(input_file, output_dir, ffmpeg_path):
    """Convert any audio file to WAV format"""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.wav")
    
    log_phase("Conversion", f"Converting {os.path.basename(input_file)} to WAV...")
    
    try:
        cmd = [
            ffmpeg_path,
            "-y", "-i", input_file,
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            output_file
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_file
    except subprocess.CalledProcessError:
        log_error(f"FFmpeg conversion failed for {os.path.basename(input_file)}")
        return None
    except Exception as e:
        log_error(f"Unexpected error during conversion: {str(e)}")
        return None

def get_audio_files(input_dir, ffmpeg_path):
    """Get all audio files, converting non-WAV files if needed"""
    supported_exts = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.mp4']
    wav_files = []
    conversion_dir = os.path.join(input_dir, "converted")
    os.makedirs(conversion_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path) or file.startswith('.'):
            continue
            
        ext = os.path.splitext(file)[1].lower()
        
        if ext == '.wav':
            wav_files.append(file_path)
        elif ext in supported_exts:
            converted = convert_to_wav(file_path, conversion_dir, ffmpeg_path)
            if converted:
                wav_files.append(converted)
            else:
                log_warning(f"Failed to convert {file}")
        else:
            log_warning(f"Skipping unsupported file: {file}")

    return wav_files

class Phase01:
    """Phase 1 - Noise reduction with DEMUCS"""
    def __init__(self, model_names=None):
        self.model_names = model_names or ["htdemucs", "mdx_extra_q"]

    def separate_sources(self, input_file, output_base_path):
        for model_name in self.model_names:
            model_output_path = os.path.join(output_base_path, model_name)
            os.makedirs(model_output_path, exist_ok=True)
            log_phase("Phase 01", f"Processing with DEMUCS ({model_name}):\n    {input_file}")
            demucs_args = ["-n", model_name, input_file, "-o", model_output_path]
            demucs_separate(demucs_args)

class Phase02:
    """Phase 2 - Audio segmentation"""
    def __init__(self, segment_duration):
        self.segment_duration = segment_duration

    def split_audio(self, input_file, output_dir):
        log_phase("Phase 02", f"Splitting {input_file} into {self.segment_duration}s segments")
        os.makedirs(output_dir, exist_ok=True)

        try:
            audio, sr = librosa.load(input_file, sr=44100, mono=True)
            total_duration = librosa.get_duration(y=audio, sr=sr)
            segment_paths = []

            for i in range(0, int(total_duration), self.segment_duration):
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                segment_path = os.path.join(output_dir, f"{base_name}_segment_{i//self.segment_duration+1}.wav")
                start = i * sr
                end = min((i + self.segment_duration) * sr, len(audio))
                sf.write(segment_path, audio[start:end], sr)
                segment_paths.append(segment_path)

            return segment_paths
        except Exception as e:
            log_error(f"Segmentation failed: {str(e)}")
            return []

class Phase03:
    """Phase 3 - Voice separation with OpenUnmix"""
    def separate_voices(self, segment_path, output_dir):
        log_phase("Phase 03", f"Separating voices from {segment_path}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            audio, sr = librosa.load(segment_path, sr=None)
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            estimates = predict.separate(audio_tensor, rate=sr)

            for instrument, audio_data in estimates.items():
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.squeeze().detach().numpy()
                output_file = os.path.join(output_dir, f"{os.path.basename(segment_path)}_{instrument}.wav")
                sf.write(output_file, audio_data.T, sr)

        except Exception as e:
            log_error(f"Voice separation failed: {str(e)}")

class Phase04:
    """Phase 4 - Quality enhancement with VoiceFixer"""
    def __init__(self, ffmpeg_path):
        self.ffmpeg_path = ffmpeg_path
        log_info("Initializing VoiceFixer...")
        self.voice_fixer = VoiceFixer()

    def enhance_audio(self, input_file, output_file):
        try:
            self.voice_fixer.restore(input_file, output_file, cuda=False)
        except Exception as e:
            log_error(f"VoiceFixer enhancement failed: {str(e)}")

class AudioProcessingPipeline:
    def __init__(self, segment_duration, ffmpeg_path):
        self.phase01 = Phase01()
        self.phase02 = Phase02(segment_duration)
        self.phase03 = Phase03()
        self.phase04 = Phase04(ffmpeg_path)

    def process_audio(self, input_file, output_base_dir):
        # Phase 1: Noise reduction
        phase01_dir = os.path.join(output_base_dir, "Phase01_Reduced_Noise")
        self.phase01.separate_sources(input_file, phase01_dir)

        # Phase 2: Segmentation
        phase02_dir = os.path.join(output_base_dir, "Phase02_Segments")
        main_output = os.path.join(phase01_dir, "htdemucs", os.path.basename(input_file))
        segments = self.phase02.split_audio(main_output, phase02_dir)
        if not segments:
            return False

        # Phase 3: Voice separation
        phase03_dir = os.path.join(output_base_dir, "Phase03_Separated_Voices")
        for segment in segments:
            self.phase03.separate_voices(segment, phase03_dir)

        # Phase 4: Enhancement
        phase04_dir = os.path.join(output_base_dir, "Phase04_Enhanced")
        os.makedirs(phase04_dir, exist_ok=True)
        for file in os.listdir(phase03_dir):
            if file.endswith(".wav"):
                input_path = os.path.join(phase03_dir, file)
                output_path = os.path.join(phase04_dir, f"enhanced_{file}")
                self.phase04.enhance_audio(input_path, output_path)
        
        return True

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help_menu()
        sys.exit(0)
    # Check FFmpeg first
    ffmpeg_path = check_ffmpeg()
    if not ffmpeg_path:
        log_error("FFmpeg not found. Please install FFmpeg first.")
        log_info("Installation instructions:")
        log_info("  macOS: brew install ffmpeg")
        log_info("  Linux: sudo apt-get install ffmpeg")
        log_info("  Windows: Download from ffmpeg.org")
        sys.exit(1)

    # Configuration
    input_dir = "source_audio"
    output_dir = "Audio_Processing_Output"
    segment_duration = 60  # seconds

    # Create input directory if needed
    os.makedirs(input_dir, exist_ok=True)

    # Get and convert audio files
    audio_files = get_audio_files(input_dir, ffmpeg_path)
    if not audio_files:
        log_error(f"No supported audio files found in {input_dir}")
        log_info(f"Supported formats: .wav, .mp3, .ogg, .flac, .aac, .m4a, .mp4")
        sys.exit(1)

    # Initialize pipeline
    pipeline = AudioProcessingPipeline(segment_duration, ffmpeg_path)
    log_info("Initializing VoiceFixer (may take several minutes first time)...")
    try:
        VoiceFixer()  # Pre-load models
    except Exception as e:
        log_error(f"VoiceFixer initialization failed: {str(e)}")
        sys.exit(1)

    # Process files
    success_count = 0
    for file in audio_files:
        log_info(f"\nProcessing: {os.path.basename(file)}")
        start_time = time.time()
        if pipeline.process_audio(file, output_dir):
            success_count += 1
            log_info(f"Completed in {time.time()-start_time:.2f} seconds")
        else:
            log_error(f"Failed to process {os.path.basename(file)}")

    log_info(f"\nProcessing complete! {success_count}/{len(audio_files)} files processed successfully")
    if success_count < len(audio_files):
        log_warning("Some files failed to process. Check logs for details.")
