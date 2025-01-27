# AI_ML_Advanced_Audio_Processing_Pipeline
Artificial Intelligence / Machine Learning - Advanced_Audio_Processing_Pipeline

# Audio Processing Pipeline

## Overview

This project provides a robust pipeline for audio processing, focusing on noise reduction, audio segmentation, voice separation, and audio enhancement. The system utilizes cutting-edge tools like **DEMUCS**, **OpenUnmix**, and **VoiceFixer** to handle complex audio processing tasks.

## Features

1. **Noise Reduction**: Utilizes DEMUCS models (e.g., `htdemucs` and `mdx_extra_q`) to reduce noise and isolate audio components.
2. **Audio Segmentation**: Splits audio into smaller segments for easier processing.
3. **Voice Separation**: Separates individual voices or instruments using OpenUnmix.
4. **Audio Enhancement**: Enhances audio quality with VoiceFixer and applies vocal excitation.

## Requirements

- Python 3.8+
- Dependencies:
  ```bash
  pip install demucs librosa soundfile torch voicefixer openunmix ffmpeg-python
  ```
- FFmpeg: Ensure FFmpeg is installed and accessible in your system's PATH.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/audio-processing-pipeline.git
   cd audio-processing-pipeline
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Input Structure

Place your audio files in the `source_audio/` directory.

### Run the Pipeline

Execute the script to process audio files:

```bash
python script_debug.py
```

### Output Structure

The output will be organized in the following directories:

- **Phase01\_Reduced\_Noise**: Noise-reduced audio files.
- **Phase02\_Segments**: Segmented audio files.
- **Phase03\_Separated\_Voices**: Audio files with separated voices.
- **Phase04\_Final\_Enhancements**: Enhanced and finalized audio files.

## Customization

### Adjusting Models

You can customize the models used in `Phase01` by editing the `model_names` list:

```python
model_names = ["htdemucs", "mdx_extra_q"]
```

### Changing Segment Duration

Modify the segment duration (in seconds) in the main script:

```python
segment_duration = 60
```

## Project Structure

```
Audio_Processing_Pipeline/
├── script_debug.py      # Main processing script
├── source_audio/        # Input audio files
├── Audio_Processing_Output/  # Processed audio outputs
├── README.md            # Documentation
└── requirements.txt     # Python dependencies
```

## Dependencies

- [DEMUCS](https://github.com/facebookresearch/demucs): Noise reduction and source separation.
- [OpenUnmix](https://github.com/sigsep/open-unmix-pytorch): Music separation.
- [VoiceFixer](https://github.com/haoheliu/voicefixer): Audio enhancement.
- [Librosa](https://librosa.org/): Audio analysis and processing.
- [SoundFile](https://pysoundfile.readthedocs.io/): Read and write audio files.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## Acknowledgments

- Thanks to the open-source community for providing the amazing tools used in this project.

