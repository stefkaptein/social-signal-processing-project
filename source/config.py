import os
import re
from typing import List

data_source_path = "../data/ICSIplus"


def get_words_file_paths(meeting_name: str) -> List[str]:
    file_paths: List[str] = []

    transcript_names_re = re.compile(f"{meeting_name}.[A-Z]?.words.xml")
    for (dir_path, _, filenames) in os.walk(f"{data_source_path}/Words"):
        for filename in filenames:
            if not transcript_names_re.match(filename):
                continue
            file_paths.append(f"{dir_path}/{filename}")
    return file_paths

def get_transcripts_file_path(meeting_name: str) -> str:
    return f"{data_source_path}/transcripts/{meeting_name}.mrt"