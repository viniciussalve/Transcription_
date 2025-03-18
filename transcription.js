import fs from 'fs';
import { pipeline } from '@huggingface/transformers';
import wavefile from 'wavefile';
import path from 'path';

/**
 * Audio transcription script that fetches an audio file, processes it,
 * and transcribes it using the Whisper model.
 * 
 * Configuration can be set via environment variables or defaults are provided.
 */

// Configuration options (can be moved to environment variables or CLI args)
const CONFIG = {
  // Model configuration
  model: process.env.TRANSCRIPTION_MODEL || 'Xenova/whisper-tiny.en',
  
  // Audio source configuration
  audioUrl: process.env.AUDIO_URL || 'https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250317%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250317T212121Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=05f5068f7630149553ec05fb32c4cb1fde29332c5a3bb836b7849ef3bb2438c6d18bedadd5f2e25b1d49ab6621d31c5c4af9763e511342acdaed89dfc8d1ec108dc0ac92b6fee547c21e2f3c65ffff311213f820b2910777fc92ce2d6b3de3d3bc0d85003b61f7f776ca8fb219750a6122b1832138b91dd1447d076bc335bca4ac0f9ed6510a29714d00c00b17cb83613f998f48cb1f6798d767454706c7fb9f72c61a2cf272f999f981d0865c52d56148188658e750225395d009e1ffb6eb464b26d021574a7e7cc9badbac84cc66a39a96310a7c4ffd7e38ab212f0171054ced2fab2712edb93f7ebc3bba6a2c1a12f8e288a4aea8e38cdd2c1336488a4f7b',

  // Transcription configuration
  chunkLengthSeconds: process.env.CHUNK_LENGTH_S || 30,
  strideLengthSeconds: process.env.STRIDE_LENGTH_S || 5,
  returnTimestamps: process.env.RETURN_TIMESTAMPS || 'word',
  
  // Output configuration
  outputDir: process.env.OUTPUT_DIR || './audios',
  outputFile: process.env.OUTPUT_FILE || 'transcription.json',
};

/**
 * Ensures the output directory exists
 * @param {string} directory - Path to the directory
 */
async function ensureDirectoryExists(directory) {
try {
    await fs.promises.mkdir(directory, { recursive: true });
    console.log(`Ensured directory exists: ${directory}`);
} catch (error) {
    throw new Error(`Failed to create directory ${directory}: ${error.message}`);
}
}

/**
 * Fetches audio from a URL and returns it as a buffer
 * @param {string} url - URL to fetch audio from
 * @returns {Promise<Buffer>} - Audio buffer
 */
async function fetchAudio(url) {
  console.log('Fetching audio file...');
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: ${response.status} ${response.statusText}`);
    }
    console.log('Audio file fetched successfully');
    return Buffer.from(await response.arrayBuffer());
  } catch (error) {
    // Add more specific error for URL expiration
    if (error.message.includes('403')) {
      throw new Error(`URL access denied. The URL may have expired: ${error.message}`);
    }
    throw new Error(`Error fetching audio: ${error.message}`);
  }
}

/**
 * Processes audio to prepare for transcription
 * @param {Buffer} buffer - Audio buffer
 * @returns {Float32Array} - Processed audio data
 */
function processAudio(buffer) {
  console.log('Processing audio...');
  try {
    // Create a WaveFile object from the buffer
    const wav = new wavefile.WaveFile(buffer);
    
    // Convert audio to 32-bit float format
    wav.toBitDepth('32f');
    
    // Convert audio to 16kHz sample rate (required by Whisper)
    wav.toSampleRate(16000);
    
    // Get the audio samples
    let audioData = wav.getSamples();

    // Handle multi-channel audio (convert to mono)
    if (Array.isArray(audioData)) {
      if (audioData.length > 1) {
        console.log('Converting stereo to mono...');
        // If we have more than one channel (e.g., stereo), convert to mono
        const SCALING_FACTOR = Math.sqrt(2);
        const monoChannel = new Float32Array(audioData[0].length);
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
    
    console.log('Audio processing complete');
    return audioData;
  } catch (error) {
    throw new Error(`Error processing audio: ${error.message}`);
  }
}

/**
 * Main function to run the transcription process
 */
async function main() {
  // Start timing the execution
  const start = performance.now();

  try {
    // Ensure output directory exists
    await ensureDirectoryExists(CONFIG.outputDir);
    
    console.log(`Loading model: ${CONFIG.model}...`);
    // Initialize the speech recognition pipeline
    const transcriber = await pipeline('automatic-speech-recognition', CONFIG.model);
    console.log('Model loaded successfully');

    // Fetch and process audio
    const buffer = await fetchAudio(CONFIG.audioUrl);
    const audioData = processAudio(buffer);

    console.log('Starting transcription...');
    // Transcribe the audio
    const output = await transcriber(audioData, {
      return_timestamps: CONFIG.returnTimestamps,
      chunk_length_s: CONFIG.chunkLengthSeconds,
      stride_length_s: CONFIG.strideLengthSeconds
    });
    console.log('Transcription complete');

    // Create full output path
    const outputPath = path.join(CONFIG.outputDir, CONFIG.outputFile);
    
    // Save the transcription output to a JSON file
    await fs.promises.writeFile(outputPath, JSON.stringify(output, null, 2));
    console.log(`Transcription saved to ${outputPath}`);

    // Calculate execution time
    const end = performance.now();
    console.log(`Execution duration: ${(end - start) / 1000} seconds`);
    
    // Log a summary of the transcription instead of the entire output
    console.log('Transcription summary:');
    console.log(`- Text length: ${output.text?.length || 0} characters`);
    console.log(`- Word count: ${output.chunks?.length || 0} words`);
    
    // Option to log full output if needed
    // console.log('Full transcription output:', output);
    
    return output;
  } catch (error) {
    console.error("ERROR:", error.message);
    const end = performance.now();
    console.log(`Failed execution duration: ${(end - start) / 1000} seconds`);
    
    // Exit with error code
    process.exit(1);
  }
}

// Run the main function
main().catch(error => {
  console.error('Unhandled error in main function:', error);
  process.exit(1);
});
