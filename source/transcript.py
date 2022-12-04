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

        # elif isinstance(node, ObjectifiedElement):
        #     non_vocal_node = node.find("NonVocalSound")
        #     if non_vocal_node is not None:
        #         row["NonVocalSound"] = non_vocal_node.attrib.get("Description")

    return pd.DataFrame(rows)


def read_full_transcript_word(path,participant):
    with open(path,"r") as file:
        xml = objectify.parse(file)
        segment_nodes = xml.xpath("//w")
        return parse_segment_nodes_word(segment_nodes,participant)

def parse_segment_nodes_word(segment_nodes: List[Element],participant):
    rows = []
    for node in segment_nodes:
        row = {
            "id": node.attrib["{http://nite.sourceforge.net/}id"],
            "StartTime": node.attrib["starttime"],
            "EndTime": node.attrib["endtime"],
            "Participant": participant,
            "c": node.attrib["c"],
            "Text": None
        }

        raw_text = node.text
        if raw_text is None:
            continue
        clean_text = raw_text.replace("\n", "").strip()
        row["Text"] = clean_text
        rows.append(row)

        # elif isinstance(node, ObjectifiedElement):
        #     non_vocal_node = node.find("NonVocalSound")
        #     if non_vocal_node is not None:
        #         row["NonVocalSound"] = non_vocal_node.attrib.get("Description")

    return pd.DataFrame(rows)
