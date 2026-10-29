# Upgrade 02: Codebook-Usage-Diagnose (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E01 (Pooling-Symmetrie).

Misst pro RVQ-Level: aktive Codes von 4096, Top-10%-Massenanteil, Entropie ueber N generierter Docs. Geringe Aktivierung auf groben Levels = Codec limitiert.

Run: cd upgrades/pooling && NCM_ROOT=<repo-root> python3 usage.py --n 16 --device cpu


## Status: documented-negative (E06/E11)

Quantix-eng/ncm-experiment-notes hat diese Frage bereits untersucht:
- **E06** (Sudoku): 15/16 Encoding-Codes aktiv bei weighted pooling, aber
  Generator-Agreement faellt auf 1.65% -> Encoding-Diagnose, kein
  Generierungs-Nutzen.
- **E11** (Kontrolle, LM1B/TinyStories): der konstante Level aus dem
  Sudoku-Fall generalisiert NICHT auf Text-Strata.

Erwartung fuer diesen Branch: geringe Code-Aktivierung auf groben Levels ist
ein Codec-Phaenomen und erklaert keine Qualitaetsluecke. Der N>=64-Lauf dient
nur der Bestaetigung, nicht der Suche nach einem Hebel.
