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

## Dependencies

- [@huggingface/transformers](https://www.npmjs.com/package/@huggingface/transformers) - For accessing the Whisper speech recognition model
- [wavefile](https://www.npmjs.com/package/wavefile) - For processing WAV audio files
- [fs](https://nodejs.org/api/fs.html) - Node.js file system module

## Usage

Run the script with:

```bash
node transcription.js
```

The script will:
1. Fetch the audio file from the specified URL
2. Process the audio to meet Whisper's requirements
3. Transcribe the audio using the Whisper tiny.en model
4. Save the transcription to `./audios/transcription.json`
5. Log the execution time and transcription output

## Configuration

The script uses the following configuration:

- Model: `Xenova/whisper-tiny.en` (a small English Whisper model)
- Chunk length: 30 seconds
- Stride length: 5 seconds
- Output includes word-level timestamps

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

## Performance

The script includes timing measurements to track execution duration, which is logged upon completion.

## Error Handling

The script includes error handling for:
- Failed audio file fetching
- Audio processing issues
- Transcription failures

## License

[Add your license information here]

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the Whisper model
- [Whisper](https://github.com/openai/whisper) by OpenAI for the speech recognition technology