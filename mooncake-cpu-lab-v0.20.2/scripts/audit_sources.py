"""Show changed upstream source files relative to the delivered baseline."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lock=json.loads((root/'UPSTREAM.lock.json').read_text())
changed=[]
for name,expected in lock['sha256'].items():
    p=root/name
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(name)
print(json.dumps({'modified_upstream_files':changed,'baseline':lock['upstream']},indent=2))
