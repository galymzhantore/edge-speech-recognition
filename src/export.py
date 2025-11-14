"""Compatibility shim for legacy workflows.

This now routes to ONNX export only. TFLite export has been removed.
"""

from .export_onnx import main

if __name__ == "__main__":
    main()
