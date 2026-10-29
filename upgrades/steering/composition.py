"""Upgrade 04b - Kompositions-Test fuer Code-Splices (Quantix E04).

E04: Doppel-Edits interferieren in ~45% der Faelle ("single-edit success
does not imply composition"). Hier: zwei Splices (Level A von Donor 1,
Level B von Donor 2) einzeln und gemeinsam anwenden und messen, ob der
gemeinsame Effekt additiv bleibt.
"""
import argparse
import torch
from common import load, gen_codes, build_prefix, load_docs

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--level-a", type=int, default=0)
ap.add_argument("--level-b", type=int, default=2)
ap.add_argument("--nslots", type=int, default=4)
ap.add_argument("--pair", type=str, default="0,1,2", help="target,donor1,donor2")
a = ap.parse_args()

CFG, tk, net, src_tok = load(device=a.device)
P = int(CFG.model.prefix_len)
PAD = int(CFG.model.prefix_pad_token_id)
EOS = tk.pre_tokenizer.eos_token_id
items = load_docs(8)
ti, d1, d2 = (int(x) for x in a.pair.split(","))

levels = [int(l) for l in tk.vqvae.vq.levels]
offs = [0]
for l in levels:
    offs.append(offs[-1] + l)

def decode(g):
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    with torch.no_grad():
        toks = torch.argmax(tk.vqvae.decoder(zl[-1]), dim=-1)[0].tolist()
        if EOS in toks:
            toks = toks[:toks.index(EOS)]
        return tk.pre_tokenizer.decode(toks, skip_special_tokens=True)

def codes_and_text(doc):
    ids = build_prefix(doc, src_tok, P, PAD)
    g = gen_codes(net, [ids], device=a.device)
    return g, decode(g)

def splice(g_target, donor_g, level):
    g = g_target.clone()
    s = offs[level]
    g[:, s:s + a.nslots] = donor_g[:, s:s + a.nslots]
    return g

gT, txtT = codes_and_text(items[ti])
g1, _ = codes_and_text(items[d1])
g2, _ = codes_and_text(items[d2])

gA = splice(gT, g1, a.level_a)
gB = splice(gT, g2, a.level_b)
gAB = splice(gA, g2, a.level_b)  # A zuerst, dann B

txtA, txtB, txtAB = decode(gA), decode(gB), decode(gAB)
print("TARGET :", txtT[:180])
print("SPLICE-A (lvl", a.level_a, ", donor", d1, "):", txtA[:180])
print("SPLICE-B (lvl", a.level_b, ", donor", d2, "):", txtB[:180])
print("JOINT  :", txtAB[:180])

def diff(x, y):
    return sum(1 for p, q in zip(x, y) if p != q) + abs(len(x) - len(y))

cA, cB, cAB = diff(txtT, txtA), diff(txtT, txtB), diff(txtT, txtAB)
over = diff(txtA, txtB)  # Ueberlappung der Einzel-Effekte
print(f"\nsingle_A_changed={cA} single_B_changed={cB} joint_changed={cAB} effect_overlap={over}")
interfere = cAB < max(cA, cB) * 0.8 or cAB > cA + cB + 0.2 * (cA + cB)
print("composition:", "INTERFERES (nicht additiv, vgl. E04)" if interfere
      else "additiv im Rahmen der Messunsicherheit")
