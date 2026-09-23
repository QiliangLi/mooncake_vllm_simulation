"""Copy original Python package from the pinned wheel for editable source execution."""
import importlib.metadata
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
dist=importlib.metadata.distribution('vllm')
if dist.version!='0.20.2': raise SystemExit('requires vllm==0.20.2')
src=Path(dist.locate_file('vllm'))
dst=root/'vendor/vllm/vllm'
if dst.exists(): raise SystemExit('Source directory already exists; refusing to overwrite your edits.')
for entry in dist.files or []:
    # Only wheel-owned files; never copy stale modules from an older install.
    rel=Path(str(entry))
    if rel.parts[0]=='vllm' and rel.suffix in ('.py','.json','.jinja','.jinja2'):
        f=Path(dist.locate_file(entry))
        out=dst/rel.relative_to('vllm'); out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,out)
print(f'Editable upstream source: {dst}')
