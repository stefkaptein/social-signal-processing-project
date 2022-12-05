from typing import List
from lxml import objectify

import pandas as pd
from lxml.etree import Element
from lxml.objectify import ObjectifiedElement, StringElement
from pandas import DataFrame


def read_full_transcript_phrase(path: str):
    with open(path,"r") as file:
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


def read_full_transcript_word(path,participant):
    with open(path,"r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//w")
        df_word = parse_segment_nodes_word(segment_nodes,participant,"w")
        segment_nodes = xml.xpath("//vocalsound")
        df_vocalsound = parse_segment_nodes_word(segment_nodes,participant,"vocalsound")
        segment_nodes = xml.xpath("//nonvocalsound")
        df_nonvocalsound = parse_segment_nodes_word(segment_nodes,participant,"nonvocalsound")
        segment_nodes = xml.xpath("//comments")
        df_comments = parse_segment_nodes_word(segment_nodes,participant,"comments")
        segment_nodes = xml.xpath("//disfmarker")
        df_disfmarker = parse_segment_nodes_word(segment_nodes,participant,"disfmarker")
        segment_nodes = xml.xpath("//pause")
        df_pause = parse_segment_nodes_word(segment_nodes,participant,"pause")
    
    df_whole = pd.concat([df_vocalsound,df_word,df_nonvocalsound,df_comments,df_disfmarker,df_pause])
    return df_whole


def parse_segment_nodes_word(segment_nodes: List[Element],participant,feature):
    rows = []
    for node in segment_nodes:
        row = []
        if feature=="w":
            row = {
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
                "c": node.attrib["c"],
            }
            if "k" in node.attrib:
                row['k']=node.attrib["k"]
            if "qut" in node.attrib:
                row['qut']=node.attrib["qut"]
            if "t" in node.attrib:
                row['t']=node.attrib["t"]
            
            raw_text = node.text
            if raw_text is None:
                continue
            clean_text = raw_text.replace("\n", "").strip()
            row["Text"] = clean_text
            
        elif (feature=="vocalsound") | (feature=="nonvocalsound") | (feature=="comment"):
            row = {
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
                "Description": node.attrib["description"],
            }
        elif (feature=="disfmarker") | (feature=="pause"):
            row = {
                "id": node.attrib["{http://nite.sourceforge.net/}id"],
                "StartTime": node.attrib["starttime"],
                "EndTime": node.attrib["endtime"],
                "Participant": participant,
            }

        rows.append(row)

    return pd.DataFrame(rows)