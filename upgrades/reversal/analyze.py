"""Upgrade 01 - Reversal analysis on XSum (adapted from Quantix-eng E05/E06).

Question: once a concept-level readout reaches its final token, does it stay?
For each of the 9 RVQ levels we decode tokens from the cumulative z_hat and
measure per-position stability and reversals (reached final token, left later).
"""
import argparse
import torch
from common import load, gen_codes, build_prefix, load_docs

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()

CFG, tk, net, src_tok = load(device=a.device)
P = int(CFG.model.prefix_len)
PAD = int(CFG.model.prefix_pad_token_id)
EOS = tk.pre_tokenizer.eos_token_id
items = load_docs(a.n)

results = []
for i, doc in enumerate(items):
    ids = build_prefix(doc, src_tok, P, PAD)
    g = gen_codes(net, [ids], device=a.device)          # (1, total_slots)
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    decs = []
    with torch.no_grad():
        for z in zl:
            toks = torch.argmax(tk.vqvae.decoder(z), dim=-1)[0].tolist()
            if EOS in toks:
                toks = toks[:toks.index(EOS)]
            decs.append(toks)
    L = len(decs)
    final = decs[-1]
    agrees = [sum(1 for x, y in zip(d, final) if x == y) / max(len(final), 1) for d in decs]
    rev = 0
    for pos in range(len(final)):
        reached = False
        for k in range(L):
            if pos >= len(decs[k]):
                continue  # coarser level ended its summary earlier here
            if decs[k][pos] == final[pos]:
                reached = True
            elif reached:
                rev += 1
                break
    results.append({"idx": i, "n_levels": L, "final_len": len(final),
                    "agree_curve": agrees, "reversals": rev})
    print(f"doc {i}: len={len(final)} agree@k=" +
          ",".join(f"{x:.2f}" for x in agrees) + f" reversals={rev}", flush=True)

revs = [r["reversals"] for r in results]
print(f"\nSUMMARY docs={len(results)} mean_reversals={sum(revs)/len(revs):.2f} "
      f"docs_with_reversals={sum(1 for r in revs if r > 0)}")
