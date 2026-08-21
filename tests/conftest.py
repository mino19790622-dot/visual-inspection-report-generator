# tests/conftest.py
"""Shared test fixtures & environment setup.

Design principle: no test ever touches the network (DashScope) or loads the
real 50MB ONNX model. External boundaries are mocked at the module boundary.
"""

import os
import sys

# allow running pytest from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# some modules construct OpenAI clients at import/init; CI has no real key
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-key-for-ci-only")
