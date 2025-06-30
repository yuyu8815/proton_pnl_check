from datetime import datetime
from pathlib import Path
import subprocess
import re
from collections import defaultdict

def run_tf_g_and_extract_fire(logfile: Path) -> list:
    cmd = f"less {logfile} | tf -g"
    try:
        result = subprocess.check_output(cmd, shell=True, text=True)
        return [line for line in result.splitlines() if "FUT_TAIFEX_TXF:" in line and "[FIRE]" in line and "[FIRE DONE]" not in line]
    except:
        return []

def extract_fire_events(date: str) -> dict:
    base_path = Path("/nfs/datafiles.optiontraderlogs") / date.replace("-", "/")
    log_5f = base_path / "capital_neutrino_txf_5f" / f"output.neutrino_txf_5f.{date.replace('-', '')}.log"
    log_6f = base_path / "capital_neutrino_txf" / f"output.neutrino_txf.{date.replace('-', '')}.log"

    fire_5f = run_tf_g_and_extract_fire(log_5f)
    fire_6f = run_tf_g_and_extract_fire(log_6f)

    return parse_fire_lines(fire_5f, "5F") + parse_fire_lines(fire_6f, "6F")

def parse_fire_lines(lines, floor):
    events = []
    for line in lines:
        m = re.search(r"FUT_TAIFEX_TXF:(\d+)", line)
        if not m:
            continue
        try:
            timestamp = datetime.strptime(line[:15], "%H:%M:%S.%f")
            events.append((m.group(1), floor, timestamp, line))
        except:
            continue
    return events
