"""Upgrade 02 - Codebook usage / pooling diagnosis on XSum (from Quantix E01).

E01 found only 15/16 encoding codes active with weighted pooling (Sudoku).
Here: per-level unique-code counts, active fraction of the 4096-entry
quantizers, and entropy over N generated docs. Low usage at coarse levels
indicates the codec (not the generator) limits diversity.
"""
import argparse
import torch
from common import load, gen_codes, build_prefix, load_docs

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

CFG, tk, net, src_tok = load(device=a.device)
P = int(CFG.model.prefix_len)
PAD = int(CFG.model.prefix_pad_token_id)
items = load_docs(a.n)

allg = []
for i, doc in enumerate(items):
    ids = build_prefix(doc, src_tok, P, PAD)
    g = gen_codes(net, [ids], device=a.device)
    allg.append(g[0])
G = torch.stack(allg)  # (N, total_slots)
levels = tk.vqvae.vq.levels
offs = [0]
for l in levels:
    offs.append(offs[-1] + int(l))
print("docs={} total_slots={} levels={}".format(G.shape[0], G.shape[1], list(levels)))
hdr = "{:>3} {:>6} {:>7} {:>12} {:>8} {:>8}".format("lvl", "slots", "unique", "active/4096", "top10%", "entropy")
print(hdr)
for li, l in enumerate(levels):
    seg = G[:, offs[li]:offs[li + 1]].reshape(-1)
    cnt = torch.bincount(seg, minlength=4096).double()
    p = cnt / cnt.sum()
    ent = -(p[p > 0] * p[p > 0].log2()).sum().item()
    top = cnt.sort(descending=True).values[: len(cnt) // 10].sum().item() / cnt.sum().item()
    uniq = int((cnt > 0).sum())
    print("{:>3} {:>6} {:>7} {:>10}/4096 {:>8.3f} {:>8.2f}".format(li, int(l), uniq, uniq, top, ent))
