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
    meeting_names = datasets = """Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017 Bmr001 Bmr002 Bmr005 Bmr007 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr018 Bmr019 Bmr021 Bmr022 Bmr024 Bmr025 Bmr026 Bmr027 Bmr029 Bns001 Bns002 Bns003 Bro003 Bro004 Bro005 Bro007 Bro008 Bro010 Bro011 Bro012 Bro013 Bro014 Bro015 Bro016 Bro017 Bro018 Bro019 Bro021 Bro022 Bro023 Bro024 Bro025 Bro026 Bro027 Bro028 Bsr001 Btr001 Btr002""".split(" ")
    for meeting_name in meeting_names:
        topic_boundaries_level_zero = extract_topic_boundaries_and_level(meeting_name, 0)
        topic_boundaries_level_one = extract_topic_boundaries_and_level(meeting_name, 1)
        topic_boundaries_level_two = extract_topic_boundaries_and_level(meeting_name, 2)

        pd.DataFrame(topic_boundaries_level_zero).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_0.csv", sep=',', index=False)
        pd.DataFrame(topic_boundaries_level_one).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_1.csv", sep=',', index=False)
        pd.DataFrame(topic_boundaries_level_two).to_csv(f"../topic_boundaries/{meeting_name}_topic_boundaries_lvl_2.csv", sep=',', index=False)