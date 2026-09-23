"""Real scheduler / simulated execution research harness."""
import sys
from pathlib import Path
source = Path(__file__).resolve().parents[1] / 'vendor/vllm'
if (source / 'vllm/__init__.py').exists(): sys.path.insert(0, str(source))
