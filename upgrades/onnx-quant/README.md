# Upgrade 03: ONNX-Weights + Quantisierung

Quelle: yrangana/ncm-lab verweist auf mrfakename/ncm-webgpu-weights (ONNX-Exporte der offiziellen NCM-Checkpoints, MIT). Das ist der einzige oeffentliche Satz offizieller Gewichte ausserhalb des Original-Repos.

Schritte:
1. ./download.sh  (laedt ONNX-Weights nach ~/ncm-upgrades/onnx-weights)
2. Quantisierung wie yrangana/tools/quantize.py (4-Bit-Build, TinyStories-Referenz: 260 MB)
3. Ziele: jev-mid GTX1070 (int4) und/oder onnxruntime-web Demo (WebGPU/wasm, COOP/COEP noetig)
