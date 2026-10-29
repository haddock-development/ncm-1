# Upgrade 01: Reversal-Analyse (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E05/E06 (LM1B/Sudoku-Reversals) auf das XSum-Checkpoint.

Frage: Sind Readouts zwischen RVQ-Levels monoton? E05/E06: nein - korrekte Zwischenstaende werden spaeter revidiert. Kann erklaeren, warum unser greedy ROUGE unter dem Paper-Wert liegt.

Run: cd upgrades/reversal && NCM_ROOT=<repo-root> python3 analyze.py --n 8 --device cpu

## Level-Persistenz (Smoke, N=4, gram=4)

Coarse-Level-Claims "verschwinden" zu ~100% im finalen Level - aber das ist
vom n-Gramm-Proxy so erwartet: grobe Levels formulieren anders und werden im
feineren Level umformuliert (Refinement, nicht Verlust). Level 7: 83% lost,
Level 8 (feinste): 0% lost by construction. Aussagekraeftig ist daher eher
der Uebergang von Level 6/7 -> final (hier: 97.6% / 83.3% lost). Fuer ein
echtes Faktualitaets-Urteil: N>=64 + gram 2-3 Sensitivitaet + optional
die echte Claim-Extraction aus ncm-hierarchy-factuality.
