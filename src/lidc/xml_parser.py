# src/lidc/xml_parser.py
"""
Parse LIDC XML files and extract ROIs/contours.

Provides a simple parser that returns contours grouped by reader and by ROI.
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# A contour point: (x, y, z) in physical coordinates
Contour = List[Tuple[float, float, float]]

def parse_lidc_xml(xml_path: str) -> Dict[str, List[Contour]]:
    """
    Parse an LIDC XML and return a mapping: reader_id -> list of contours.
    Each contour is a list of (x,y,z) points (physical coords).
    Note: returns only 'edgeMap' contours under 'unblindedReadNodule' inside 'readingSession'.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    per_reader = {}  # reader_id -> list of contours

    # Try reading sessions (newer schema)
    reading_sessions = root.findall(".//readingSession")
    if reading_sessions:
        for idx, rs in enumerate(reading_sessions):
            rid = rs.attrib.get("id") or rs.attrib.get("reader") or f"reader_{idx}"
            contours = []
            for unblinded in rs.findall("unblindedReadNodule"):
                for roi in unblinded.findall("roi"):
                    pts = []
                    for edge in roi.findall("edgeMap"):
                        # Some XMLs have xCoord,yCoord,zCoord tags
                        x = edge.find("xCoord")
                        y = edge.find("yCoord")
                        z = edge.find("zCoord")
                        if x is None or y is None or z is None:
                            continue
                        pts.append((float(x.text), float(y.text), float(z.text)))
                    if len(pts) >= 3:
                        contours.append(pts)
            if contours:
                per_reader[rid] = contours

    # Fallback: try older schema - search all unblindedReadNodule anywhere
    if not per_reader:
        contours = []
        for unblinded in root.findall(".//unblindedReadNodule"):
            for roi in unblinded.findall("roi"):
                pts = []
                for edge in roi.findall("edgeMap"):
                    x = edge.find("xCoord")
                    y = edge.find("yCoord")
                    z = edge.find("zCoord")
                    if x is None or y is None or z is None:
                        continue
                    pts.append((float(x.text), float(y.text), float(z.text)))
                if len(pts) >= 3:
                    contours.append(pts)
        if contours:
            per_reader["reader_0"] = contours

    return per_reader
