"""Run existing pixel diagnostics without overwriting previous evidence."""
from pathlib import Path
import json


def write_new_json(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
