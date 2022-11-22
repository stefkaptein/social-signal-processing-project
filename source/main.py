from typing import List
import source.config as cfg

from source.transcript import read_full_transcript

meeting_name = "Bdb001"

if __name__ == "__main__":
    words_path: List[str] = cfg.get_words_file_paths(meeting_name)
    transcript_path: str = cfg.get_transcripts_file_path(meeting_name)

    read_full_transcript(transcript_path)
