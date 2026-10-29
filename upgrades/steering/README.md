# Upgrade 04: Concept-Code-Steering (XSum)

Adaptiert aus Quantix-eng/ncm-experiment-notes E02/E04 (Terminal-Code-Editing am Sudoku-Checkpoint).

Spliced Codeslots eines Spender-Dokuments in die Generierung eines Ziel-Dokuments (Level + Slotanzahl waehlbar) und misst, wie stark der decodierte Text reagiert. Erwartung laut E02/E04: partielles Steering, Interferenz bei mehreren Edits.

Run: cd upgrades/steering && NCM_ROOT=<repo-root> python3 edit_codes.py --level 0 --nslots 4 --pair 0,1


## Kompositions-Test (E04-Lektion)

composition.py wendet zwei Splices (unterschiedliche Levels, unterschiedliche
Donoren) einzeln und gemeinsam an. E04 warnte: Doppel-Edits interferieren in
~45%. Der Test klassifiziert, ob unser Slot-Splice additiv komponiert.

Run: cd upgrades/steering && python3 composition.py --level-a 0 --level-b 2 --pair 0,1,2
