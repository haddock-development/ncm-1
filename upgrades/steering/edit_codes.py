"""Upgrade 04 - Concept-code editing / steering on XSum (from Quantix E02/E04).

E02/E04 (Sudoku): terminal codes can be read and overwritten; ~76% of single
edits land, but two edits interfere in ~45% of cases. Here: splice code slots
from a donor document into a target generation and observe how the decoded
summary changes. Expectations: partial steering, interference likely.
"""
import argparse
import torch
from common import load, gen_codes, build_prefix, load_docs

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--level", type=int, default=0, help="RVQ level to splice (0=coarsest)")
ap.add_argument("--nslots", type=int, default=4, help="how many slots of that level to splice")
ap.add_argument("--pair", type=str, default="0,1", help="target,donor doc indices")
a = ap.parse_args()

CFG, tk, net, src_tok = load(device=a.device)
P = int(CFG.model.prefix_len)
PAD = int(CFG.model.prefix_pad_token_id)
EOS = tk.pre_tokenizer.eos_token_id
items = load_docs(8)
ti, di = (int(x) for x in a.pair.split(","))

def codes_and_text(doc):
    ids = build_prefix(doc, src_tok, P, PAD)
    g = gen_codes(net, [ids], device=a.device)
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    with torch.no_grad():
        toks = torch.argmax(tk.vqvae.decoder(zl[-1]), dim=-1)[0].tolist()
        if EOS in toks:
            toks = toks[:toks.index(EOS)]
        text = tk.pre_tokenizer.decode(toks, skip_special_tokens=True)
    return g, text

gT, txtT = codes_and_text(items[ti])
gD, txtD = codes_and_text(items[di])
print("TARGET  :", txtT[:220])
print("DONOR   :", txtD[:220])

levels = [int(l) for l in tk.vqvae.vq.levels]
offs = [0]
for l in levels:
    offs.append(offs[-1] + l)
start = offs[a.level]
gS = gT.clone()
gS[:, start:start + a.nslots] = gD[:, start:start + a.nslots]
_, zl = tk.vqvae.vq.idx_to_zhats(gS, return_zlevels=True)
with torch.no_grad():
    toks = torch.argmax(tk.vqvae.decoder(zl[-1]), dim=-1)[0].tolist()
    if EOS in toks:
        toks = toks[:toks.index(EOS)]
    txtS = tk.pre_tokenizer.decode(toks, skip_special_tokens=True)
print("SPLICED :", txtS[:220])
changed = sum(1 for x, y in zip(txtS, txtT) if x != y) + abs(len(txtS) - len(txtT))
print(f"changed_chars={changed} level={a.level} nslots={a.nslots}")
