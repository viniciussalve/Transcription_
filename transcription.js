import fs from 'fs';
import { pipeline } from '@huggingface/transformers';
import wavefile from 'wavefile';

// Start timing the execution
let start = performance.now();

// Initialize the speech recognition pipeline with a small English Whisper model
let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');

// URL to the audio file we want to transcribe
let url = 'https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250317%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250317T212121Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=05f5068f7630149553ec05fb32c4cb1fde29332c5a3bb836b7849ef3bb2438c6d18bedadd5f2e25b1d49ab6621d31c5c4af9763e511342acdaed89dfc8d1ec108dc0ac92b6fee547c21e2f3c65ffff311213f820b2910777fc92ce2d6b3de3d3bc0d85003b61f7f776ca8fb219750a6122b1832138b91dd1447d076bc335bca4ac0f9ed6510a29714d00c00b17cb83613f998f48cb1f6798d767454706c7fb9f72c61a2cf272f999f981d0865c52d56148188658e750225395d009e1ffb6eb464b26d021574a7e7cc9badbac84cc66a39a96310a7c4ffd7e38ab212f0171054ced2fab2712edb93f7ebc3bba6a2c1a12f8e288a4aea8e38cdd2c1336488a4f7b';

try {
    // Fetch the audio file and convert it to a Buffer
    let response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
    let buffer = Buffer.from(await response.arrayBuffer());

    // Create a WaveFile object from the buffer
    let wav = new wavefile.WaveFile(buffer);
    // Convert audio to 32-bit float format
    wav.toBitDepth('32f');
    // Convert audio to 16kHz sample rate (required by Whisper)
    wav.toSampleRate(16000);
    // Get the audio samples
    let audioData = wav.getSamples();

    // Handle multi-channel audio (convert to mono)
    if (Array.isArray(audioData)) {
        if (audioData.length > 1) {
            // If we have more than one channel (e.g., stereo), convert to mono
            let SCALING_FACTOR = Math.sqrt(2);
            let monoChannel = new Float32Array(audioData[0].length);
            for (let i = 0; i < audioData[0].length; ++i) {
                // Average the channels and apply scaling factor
                monoChannel[i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
            }
            audioData = monoChannel;
        } else {
            // Use only the first channel
            audioData = audioData[0];
        }
    }

    // Transcribe the audio with word-level timestamps
    // Using chunking (30s) and stride (5s) to process longer audio files
    let output = await transcriber(audioData, {
        return_timestamps: 'word',
        chunk_length_s: 30,
        stride_length_s: 5
    });

    // Save the transcription output to a JSON file
    fs.promises.writeFile("./audios/transcription.json", JSON.stringify(output))
        .then(() => console.log("done."))
        .catch(err => console.error("Error writing file:", err));

    // Calculate execution time
    let end = performance.now();
    // Log the execution time in seconds
    console.log(`Execution duration: ${(end - start) / 1000} seconds`);
    // Log the transcription output
    console.log(output);
} catch (error) {
    console.error("Error processing audio:", error);
    let end = performance.now();
    console.log(`Failed execution duration: ${(end - start) / 1000} seconds`);
}