# Upgrade 01: Reversal-Analyse (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E05/E06 (LM1B/Sudoku-Reversals) auf das XSum-Checkpoint.

Frage: Sind Readouts zwischen RVQ-Levels monoton? E05/E06: nein - korrekte Zwischenstaende werden spaeter revidiert. Kann erklaeren, warum unser greedy ROUGE unter dem Paper-Wert liegt.

Run: cd upgrades/reversal && NCM_ROOT=<repo-root> python3 analyze.py --n 8 --device cpu
