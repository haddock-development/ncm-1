# Upgrade 04: Concept-Code-Steering (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E02/E04 (Terminal-Code-Editing am Sudoku-Checkpoint).

Spliced Codeslots eines Spender-Dokuments in die Generierung eines Ziel-Dokuments (Level + Slotanzahl waehlbar) und misst, wie stark der decodierte Text reagiert. Erwartung laut E02/E04: partielles Steering, Interferenz bei mehreren Edits.

Run: cd upgrades/steering && NCM_ROOT=<repo-root> python3 edit_codes.py --level 0 --nslots 4 --pair 0,1
