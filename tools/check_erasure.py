#!/usr/bin/env python3
"""Verify the exercised disabled marker kernel has no observation work in PTX."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ptx",type=Path)
args = parser.parse_args()
source = args.ptx.read_text()
match = re.search(r"\.entry\s+\S*disabled_probe\S*\s*\([^)]*\)\s*(?:\.\w+[^\n]*\n\s*)*\{(.*?)^\}",source,re.S|re.M)
body = match[1] if match else ""
# The live probe stores a constant canary. Any load, call, timer, collective,
# atomic or branch indicates work beyond that disabled-path contract.
forbidden = re.findall(r"globaltimer|\b(?:ld\.|call\b|atom\.|red\.|bar\.|bra\b)",body)
passed = bool(body) and "st.global.u32" in body and not forbidden
print(json.dumps({"scope":"disabled marker PTX erasure, no GPU execution", "path":str(args.ptx.resolve()),
                  "sha256":hashlib.sha256(args.ptx.read_bytes()).hexdigest(),"passed":passed,
                  "forbidden":forbidden,"kernel_body":body},indent=2))
sys.exit(0 if passed else 1)
