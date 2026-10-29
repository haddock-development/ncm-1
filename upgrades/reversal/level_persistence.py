"""Upgrade 01b - Level-Persistenz (fakt-nahe Reversal-Messung, vgl.
aadithya-vimal/ncm-hierarchy-factuality und Quantix E05/E06).

Statt nur token-Positionen zu zaehlen, decodiert dieses Skript ALLE kumulativen
Level und misst persistenz auf Claim-Niveau (Satz-8-gram-Proxy):
- Ein Claim gilt als "entstanden", wenn er erstmals auf Level k auftaucht.
- "Reversal": Claim taucht auf Level k auf, fehlt aber im finalen Level.
Das misst, ob inhaltliche Aussagen (nicht nur Tokens) zwischen Levels
verschwinden - die offen gestellte Groesserevaluation des Reversal-Branches.
"""
import argparse
import re
import torch
from common import load, gen_codes, build_prefix, load_docs

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16, help="Anzahl Docs (N>=64 empfohlen)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--gram", type=int, default=4, help="n fuer Claim-n-Gramme")
a = ap.parse_args()

CFG, tk, net, src_tok = load(device=a.device)
P = int(CFG.model.prefix_len)
PAD = int(CFG.model.prefix_pad_token_id)
EOS = tk.pre_tokenizer.eos_token_id

def norm_grams(text, n):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

born = {}      # level -> Anzahl erstmalig entstandener Claims
lost = {}      # level -> Claims, die auf level entstanden, aber im finalen fehlen
final_sizes = []

for i in range(a.n):
    doc = load_docs(i + 1)[i]
    ids = build_prefix(doc, src_tok, P, PAD)
    g = gen_codes(net, [ids], device=a.device)
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    texts = []
    with torch.no_grad():
        for z in zl:
            toks = torch.argmax(tk.vqvae.decoder(z), dim=-1)[0].tolist()
            if EOS in toks:
                toks = toks[:toks.index(EOS)]
            texts.append(tk.pre_tokenizer.decode(toks, skip_special_tokens=True))
    L = len(texts)
    grams = [norm_grams(t, a.gram) for t in texts]
    final = grams[-1]
    final_sizes.append(len(final))
    seen = set()
    for k in range(L):
        new = grams[k] - seen
        born[k] = born.get(k, 0) + len(new)
        lost[k] = lost.get(k, 0) + len(new - final)
        seen |= grams[k]
    print(f"doc {i}: final_claims={len(final)}", flush=True)

nlev = len(born)
print(f"\nSUMMARY n={a.n} gram={a.gram} mean_final_claims={sum(final_sizes)/len(final_sizes):.1f}")
print("lvl  born  lost  lost_ratio")
for k in sorted(born):
    b, l = born[k], lost[k]
    r = l / b if b else 0.0
    print(f"{k:>3} {b:>6} {l:>5} {r:>10.2%}")
tot_b, tot_l = sum(born.values()), sum(lost.values())
print(f"TOTAL born={tot_b} lost={tot_l} overall_lost_ratio={tot_l/max(tot_b,1):.2%}")
