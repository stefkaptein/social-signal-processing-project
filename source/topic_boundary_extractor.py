import xml.etree.ElementTree as ET

import pandas as pd

import data

def extract_topic_level(level, max_level, topic_iter):
    if max_level == level:
        return [topic_iter]
    first_segments = [topic_iter]
    for topic in topic_iter.findall('topic'):
        first_segments.extend(extract_topic_level(level + 1, max_level, topic))
    return first_segments


def extract_topic_boundaries_and_level(meeting_name: str, max_level: int):
    tree = ET.parse(f"../data/ICSIplus/Contributions/TopicSegmentation/{meeting_name}.topic.xml")
    root = tree.getroot()
    topic_boundaries = []
    topics = []
    for topic in root.findall('topic'):
        topics.extend(extract_topic_level(0, max_level, topic))

    for topic in topics:
        if len(topic) <= 0 or topic[0].tag == 'topic':
            continue
        topic_boundaries.append({
            'topic_id': topic.attrib['{http://nite.sourceforge.net/}id'],
            'start_segment_id': topic[0].attrib['href'].split("..")[0].split("(")[1].split(")")[0].replace('"', '')
        })
    return topic_boundaries


if __name__ == '__main__':
    meeting_names = data.all()
    for meeting_name in meeting_names:
        topic_boundaries_level_zero = extract_topic_boundaries_and_level(meeting_name, 0)
        topic_boundaries_level_one = extract_topic_boundaries_and_level(meeting_name, 1)
        topic_boundaries_level_two = extract_topic_boundaries_and_level(meeting_name, 2)

        pd.DataFrame(topic_boundaries_level_zero).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_0.csv", sep=',', index=False)
        pd.DataFrame(topic_boundaries_level_one).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_1.csv", sep=',', index=False)
        pd.DataFrame(topic_boundaries_level_two).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_2.csv", sep=',', index=False)