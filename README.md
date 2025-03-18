# Audio Transcription Tool

A Node.js application that transcribes audio files using Hugging Face's Whisper model.

## Overview

This tool fetches an audio file from a URL, processes it to meet the requirements of the Whisper speech recognition model, transcribes the audio with word-level timestamps, and saves the transcription as a JSON file.

## Features

- Audio transcription using Hugging Face's Whisper model
- Support for processing WAV files
- Automatic conversion to mono audio
- Resampling to 16kHz (required by Whisper)
- Word-level timestamps in transcription output
- Chunking support for longer audio files
- Configurable via environment variables

## Prerequisites

- Node.js (v14 or higher recommended)
- npm or yarn

## Installation

1. Clone this repository
2. Install dependencies:

```bash
npm install
# or
yarn install
```

Note: This project uses ES modules. The package.json includes `"type": "module"`.

## Dependencies

- [@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers) (v3.4.0) - For accessing the Whisper speech recognition model
- [wavefile](https://www.npmjs.com/package/wavefile) (v11.0.0) - For processing WAV audio files
- Node.js built-in modules: fs, path

## Usage

Run the script with:

```bash
node transcription.js
```

The script will:
1. Fetch the audio file from the specified URL
2. Process the audio to meet Whisper's requirements
3. Transcribe the audio using the Whisper model
4. Save the transcription to the configured output location (default: `./audios/transcription.json`)
5. Log the execution time and transcription summary

## Configuration

The script can be configured using environment variables:

| Environment Variable    | Default Value                | Description                                  |
|------------------------|------------------------------|----------------------------------------------|
| TRANSCRIPTION_MODEL    | Xenova/whisper-tiny.en       | Which Whisper model to use                   |
| AUDIO_URL              | (Default sample URL)         | URL of the audio file to transcribe          |
| CHUNK_LENGTH_S         | 30                           | Audio chunk length in seconds                |
| STRIDE_LENGTH_S        | 5                            | Stride length between chunks in seconds      |
| RETURN_TIMESTAMPS      | word                         | Timestamp granularity (word or chunk)        |
| OUTPUT_DIR             | ./audios                     | Directory to save the transcription          |
| OUTPUT_FILE            | transcription.json           | Filename for the transcription output        |

Example of running with environment variables:

```bash
TRANSCRIPTION_MODEL="Xenova/whisper-small.en" AUDIO_URL="https://example.com/audio.wav" node transcription.js
```

## Output Format

The transcription is saved as a JSON file with the following structure:

```json
{
  "text": "The complete transcription text",
  "chunks": [
    {
      "timestamp": [start_time, end_time],
      "text": "chunk text"
    },
    ...
  ]
}
```

## Error Handling

The script includes robust error handling for:
- Failed audio file fetching (including URL expiration detection)
- Audio processing issues
- Directory creation failures
- Transcription failures

## Performance

The script includes timing measurements to track execution duration, which is logged upon completion.

## License

[Add your license information here]

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the Whisper model
- [Whisper](https://github.com/openai/whisper) by OpenAI for the speech recognition technology
