from typing import List
from lxml import objectify

import pandas as pd
from lxml.etree import Element


def read_full_transcript_phrase(path: str):
    with open(path, "r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//Segment")
        return parse_segment_nodes_phrase(segment_nodes)


def parse_segment_nodes_phrase(segment_nodes: List[Element]):
    rows = []
    for node in segment_nodes:
        row = {
            "StartTime": node.attrib["StartTime"],
            "EndTime": node.attrib["EndTime"],
            "Participant": node.attrib.get("Participant"),
            "Text": None
        }

        raw_text = node.text
        if raw_text is None:
            continue
        clean_text = raw_text.replace("\n", "").strip()
        row["Text"] = clean_text
        rows.append(row)

    return pd.DataFrame(rows)


def read_full_transcript_word(path, participant):
    with open(path, "r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//w")
        df_word = parse_segment_nodes_word(segment_nodes, participant, "w")
        segment_nodes = xml.xpath("//vocalsound")
        df_vocalsound = parse_segment_nodes_word(segment_nodes, participant, "vocalsound")
        segment_nodes = xml.xpath("//nonvocalsound")
        df_nonvocalsound = parse_segment_nodes_word(segment_nodes, participant, "nonvocalsound")
        segment_nodes = xml.xpath("//comment")
        df_comments = parse_segment_nodes_word(segment_nodes, participant, "comment")
        segment_nodes = xml.xpath("//disfmarker")
        df_disfmarker = parse_segment_nodes_word(segment_nodes, participant, "disfmarker")
        segment_nodes = xml.xpath("//pause")
        df_pause = parse_segment_nodes_word(segment_nodes, participant, "pause")

    df_whole = pd.concat([df_vocalsound, df_word, df_nonvocalsound, df_comments, df_disfmarker, df_pause])
    if not df_whole.empty:
        df_whole.set_index('index', inplace=True)
        df_whole.sort_index(ascending=True, inplace=True)
        return df_whole
    return 0


def parse_segment_nodes_word(segment_nodes: List[Element], participant, feature):
    rows = []
    for node in segment_nodes:
        row = []
        if feature == "w":
            row = {
                "index": node.sourceline - 2,
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
                "c": node.attrib["c"],
            }
            if "k" in node.attrib:
                row['k'] = node.attrib["k"]
            if "qut" in node.attrib:
                row['qut'] = node.attrib["qut"]
            if "t" in node.attrib:
                row['t'] = node.attrib["t"]

            raw_text = node.text
            if raw_text is None:
                continue
            clean_text = raw_text.replace("\n", "").strip()
            row["Text"] = clean_text

        elif (feature == "vocalsound") or (feature == "nonvocalsound") or (feature == "comment"):
            row = {
                "index": node.sourceline - 2,
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
                "Description": node.attrib["description"],
            }
        elif (feature == "disfmarker") or (feature == "pause"):
            row = {
                "index": node.sourceline - 2,
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
            }
        rows.append(row)

    return pd.DataFrame(rows)


def read_full_transcript_prosody(path, participant):
    with open(path, "r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//prosody")
        df_prosody = parse_segment_nodes_prosody(segment_nodes, participant)
        segment_nodes = xml.xpath("//prosody/*")
        df_prosody = parse_segment_nodes_children(segment_nodes, participant, df_prosody)

    df_prosody.set_index('index', inplace=True)
    df_prosody.sort_index(ascending=True, inplace=True)
    return df_prosody


def parse_segment_nodes_prosody(segment_nodes: List[Element], participant):
    rows = []
    for node in segment_nodes:
        row = []
        row = {
            "index": node.sourceline - 2,
            "id": node.attrib["{http://nite.sourceforge.net/}id"],
            "f0_mean": node.attrib["f0_mean"],
            "f0_std": node.attrib["f0_std"],
            "duration": node.attrib["duration"],
            "energy": node.attrib["energy"],
            "tfidf": node.attrib["tfidf"],
            "Participant": participant,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def parse_segment_nodes_children(segment_nodes: List[Element], participant, df_prosody):
    df_prosody['words_id'] = None
    counter = 0
    for node in segment_nodes:
        id = node.attrib["href"].split('#')[1]
        df_prosody['words_id'].loc[counter] = id
        counter += 1
    return df_prosody


def read_full_transcript_segment(path: str, participant):
    with open(path, "r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//segment")
        df_segs = parse_segment_nodes_segment(segment_nodes, participant, "segment", None)
        segment_nodes = xml.xpath("//segment/*")
        df_word_ids = parse_segment_nodes_segment(segment_nodes, participant, "nite:child", df_segs)
        return df_word_ids


def parse_segment_nodes_segment(segment_nodes: List[Element], participant, feature, segments):
    if feature == "segment":
        rows = []
    else:
        segments['words_id'] = None

    counter = 0
    for node in segment_nodes:
        if feature == "segment":
            if "type" in node.attrib:
                if node.attrib['type']!='supersegment':
                    row = {
                        "id": node.attrib["{http://nite.sourceforge.net/}id"],
                        "StartTime": node.attrib["starttime"],
                        "EndTime": node.attrib["endtime"],
                        "Participant1": node.attrib["participant"],
                        "Participant2": participant,
                    }
                    if "timing-provenance" in node.attrib:
                        row['Timing Provenance'] = node.attrib["timing-provenance"]
                    rows.append(row)
            else:
                row = {
                    "id": node.attrib["{http://nite.sourceforge.net/}id"],
                    "StartTime": node.attrib["starttime"],
                    "EndTime": node.attrib["endtime"],
                    "Participant1": node.attrib["participant"],
                    "Participant2": participant,
                }
                if "timing-provenance" in node.attrib:
                    row['Timing Provenance'] = node.attrib["timing-provenance"]
                rows.append(row)
        elif feature == "nite:child":
            if "type" in node.attrib:
                if node.attrib['type']!='subsegment':
                    segments['words_id'].loc[counter] = node.attrib["href"]
                    counter += 1
            else:
                segments['words_id'].loc[counter] = node.attrib["href"]
                counter += 1
                
    if feature == "segment":
        return pd.DataFrame(rows)

    return segments


def read_full_transcript_topic_segments(path):
    lines = open(path, 'r').readlines()[2:]
    all_segments = []

    curr_seg_id = None
    prev_space = 0
    lvl_counter = 0
    is_new_topic = True
    for line in lines:
        curr_space = len(line) - len(line.lstrip(' '))
        if "topic" in line and not "/topic" in line:

            new_topic_id = line.split(" ")[curr_space + 1]
            new_topic_id = new_topic_id[9:len(new_topic_id) - 1]

            if curr_space > prev_space:
                lvl_counter += 1
            elif curr_space < prev_space:
                lvl_counter -= 1

            all_segments.append([new_topic_id, lvl_counter, None, None, False])

            prev_space = curr_space
            is_new_topic = True
        elif not "topic" in line and not "root" in line:
            curr_seg_id = line.split(" ")[curr_space + 1]
            curr_seg_ids = curr_seg_id[24:len(curr_seg_id) - 4].split("..")
            if is_new_topic:
                curr_seg_id = curr_seg_ids[0][3:len(curr_seg_ids[0]) - 1]
                all_segments[len(all_segments) - 1][2] = curr_seg_id
                is_new_topic = False
            else:
                curr_seg_id = curr_seg_ids[0][3:len(curr_seg_ids[0]) - 1] if len(curr_seg_ids) == 1 else curr_seg_ids[
                                                                                                             1][3:len(
                    curr_seg_ids[1]) - 1]
        elif "/topic" in line:
            for seg in all_segments:
                if seg[1] * 3 >= curr_space and not seg[4]:
                    seg[3] = curr_seg_id
                    if seg[1] * 3 == curr_space:
                        seg[4] = True

    [j.pop(4) for j in all_segments]
    df_topic_segments = pd.DataFrame(data=all_segments,
                                     columns=['Topic_id', 'Level', 'First Segment id', 'Last Segment id'])
    return df_topic_segments


def new_topic(curr_segments):
    first_seg = None
    last_seg = None
    if len(curr_segments) != 0:
        first_seg = curr_segments[0]
        last_seg = curr_segments[len(curr_segments) - 1]

    return first_seg, last_seg, curr_segments


def parse_segment_nodes_topic_segments(segment_nodes: List[Element], feature, segments):
    if feature == "topic":
        rows = []
    else:
        segments['Segment1_id'] = None
        segments['Segment2_id'] = None

        counter = 0
        topic_seg_counter = 0
        next_seg = segments['Line'].loc[topic_seg_counter + 1]
        first_seg_id = segment_nodes[counter].attrib.values()[0]
        last_seg_id = None

    rows = []
    i = 0
    while i < len(segment_nodes):
        node = segment_nodes[i]
        if feature == "topic":
            row = {
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "Description": node.attrib["description"],
                "Line": node.sourceline
            }
            rows.append(row)
        elif feature == "nite:child":
            seg_line = node.sourceline

            if seg_line > next_seg:
                last_seg_id = segment_nodes[i - 2].attrib.values()[0]
                # print(first_seg_id, last_seg_id)

                topic_seg_counter += 1
                next_seg = segments['Line'].loc[topic_seg_counter + 1]
                first_seg_id = node.attrib.values()[0]
        i += 1

    if feature == "topic":
        return pd.DataFrame(rows)

    return segments
