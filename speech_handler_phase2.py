import os
import time
from pathlib import Path
from faster_whisper import WhisperModel

_whisper_model = None

def get_whisper_model(model_size: str = "base") -> WhisperModel:

    #Load Whisper model (singleton — loaded once, reused across calls).
    #compute_type="int8" quantizes the model to 8-bit integers.
    #This halves memory usage with minimal accuracy loss.
    #Perfect for CPU inference.

    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper '{model_size}' model (first run downloads weights)...")
        _whisper_model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="int8"   # 8-bit quantization for faster CPU inference
        )
        print("Whisper model loaded")
    return _whisper_model


def transcribe_audio(audio_path:str, model_size:str = 'base', language:str = 'en') -> dict:

    """
    Transcribe an audio file to text using faster-whisper.
 
    Supported formats: mp3, wav, m4a, flac, ogg, mp4, webm
    (PyAV handles format conversion internally — no ffmpeg install needed)
 
    Returns a dict with:
        text        — full transcription string (used as RAG query)
        language    — detected language code
        duration_s  — audio duration in seconds
        segments    — list of timestamped segments (useful for debugging)
 
    WHY SEGMENTS?
    For long audio (recorded lectures, conference talks), segments let you
    see exactly which part of the audio produced which transcription.
    In Phase 4's UI we can highlight the relevant audio timestamp.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    #Loads the weights to cpu/gpu - one time work
    model = get_whisper_model(model_size="base") #defined above 

    print(f"Transcribing: {Path(audio_path).name}")
    start = time.time()

    #NOTE: We are just initializing the model here, the actual transcription happens in the for loop below...
    #because, the model gives a generator object(segments) that generates segments as we loop through.
    #This is done to prevent loading all the audio to the memory, save memory, 1 segment loaded at a time
    #and the transcribes are generated as the loop goes through giving our result without waiting for the loop to finish.
    segments, info = model.transcribe(
        audio_path,
        language=language,
        #higher beam -> slower, more computation. Finds the 5 words with highest sequential probabiilty to the next word and 
        #beam = 1 -> just pick the word with highest probabilty, only 1 story line, high risk of hallucination.
        #if beam = 5 -> the model holds 5 different story lines in the short term memory, as it keeps hearing words and at the end 
        #eg. ICU vs I see you in a sentence...
        #chooses the sentence(chunk) that has highest probablity of being an actual sentence.
        # re-evaluates as the word is being heard and chooses the one that gets the highest score. 
        beam_size=5, 
        vad_filter=True,       # Voice Activity Detection: skip silence, breathing noise
        vad_parameters=dict(
            min_silence_duration_ms=500   # ignore gaps < 500ms
        )
    )

    segment_list = []
    full_text_parts = []

    #Looping through the generator, transcribing as we go...
    for segment in segments:

        #store the packets of info with timestamp
        segment_list.append({
            "start": round(segment.start, 2),
            "end":   round(segment.end, 2),
            "text":  segment.text.strip()
        })

        #store just the text separately...
        full_text_parts.append(segment.text.strip())
    
    #Construct the full text by joining the segment text
    full_text = " ".join(full_text_parts)

    #compute total time elapsed for the transcription
    elapsed   = round(time.time() - start, 2)

    print(f"Transcribed in {elapsed}s")
    print(f"Text: \"{full_text[:100]}{'...' if len(full_text) > 100 else ''}\"")
    print(f"Language detected: {info.language} (confidence: {info.language_probability:.2f})")
 
    return {
        "text":       full_text,
        #info contains metadata, and is computed as we initialize the model.transcribe() itself.
        "language":   info.language,
        "duration_s": round(info.duration, 2),
        "segments":   segment_list,
        "source":     audio_path
    }


#Validate Transcription Quality:
def validate_transription_quality(transcription : dict, min_words:int = 3):

    text = transcription.get('text', '')
    words = text.split()
    
    #NOTE: we are validating the whole transcription not each segment here, so each segment is still allowed to have
    #single words like, HELLO! or YES!
    if len(words)<min_words:
        print(f"This transcription is too short short {len(words)} words, probably noise...")
        return False
    return True


def speech_to_query(audio_path:str, model_size:str = 'base') -> str|None:

    transcription = transcribe_audio(audio_path=audio_path)

    if not validate_transription_quality(transcription):
        return None
   
    return transcription['text']


if __name__ == '__main__':

    import sys

    if len(sys.argv)<2:
        print("Enter a valid path for the audio file...")
        sys.exit(1)

    audio_path = sys.argv[1]

    query = speech_to_query(audio_path=audio_path)

    if query:
        print(f" Query ready for RAG pipeline:")
        print(f'   "{query}"')
    else:
        print("\nCould not extract a valid query from the audio.")

        
    