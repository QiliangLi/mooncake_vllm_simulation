"""Copy original Python package from the pinned wheel for editable source execution."""
import importlib.metadata
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
dist=importlib.metadata.distribution('vllm')
if dist.version!='0.11.0': raise SystemExit('requires vllm==0.11.0')
src=Path(dist.locate_file('vllm'))
dst=root/'vendor/vllm/vllm'
if dst.exists(): raise SystemExit('Source directory already exists; refusing to overwrite your edits.')
for f in src.rglob('*'):
    if f.is_file() and f.suffix in ('.py','.json','.jinja','.jinja2'):
        out=dst/f.relative_to(src); out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,out)
print(f'Editable upstream source: {dst}')
