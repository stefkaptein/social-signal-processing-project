from typing import List
from lxml import objectify

import pandas as pd
from lxml.etree import Element
from lxml.objectify import ObjectifiedElement, StringElement
from pandas import DataFrame


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
    df_whole.set_index('index', inplace=True)
    df_whole.sort_index(ascending=True, inplace=True)
    return df_whole


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
            row = {
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant1": node.attrib["participant"],
                "Participant2": participant,
                "Timing Provenance": node.attrib["timing-provenance"],
            }
            rows.append(row)
        if feature == "nite:child":
            segments['words_id'].loc[counter] = node.attrib["href"]
            counter += 1
    if feature == "segment":
        return pd.DataFrame(rows)

    return segments
