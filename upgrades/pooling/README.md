# Upgrade 02: Codebook-Usage-Diagnose (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E01 (Pooling-Symmetrie).

Misst pro RVQ-Level: aktive Codes von 4096, Top-10%-Massenanteil, Entropie ueber N generierter Docs. Geringe Aktivierung auf groben Levels = Codec limitiert.

Run: cd upgrades/pooling && NCM_ROOT=<repo-root> python3 usage.py --n 16 --device cpu
