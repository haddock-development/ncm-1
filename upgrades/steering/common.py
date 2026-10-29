import os, sys, pickle, torch
from omegaconf import OmegaConf

ROOT = os.environ.get("NCM_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CK = os.environ.get("NCM_CK", os.path.expanduser("~/ncm-docker/artifacts/xsum"))
DD = os.environ.get("NCM_DATA", os.path.expanduser("~/ncm-docker/data/xsum"))


def load(task_cfg="xsum.yaml", device="cpu"):
    CFG = OmegaConf.load(os.path.join(ROOT, "config", "eval", task_cfg))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(device).eval()
    sd = torch.load(os.path.join(CK, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
    return CFG, tk, net, src_tok


@torch.no_grad()
def gen_codes(net, prefix_ids, temp=1.0, device="cpu"):
    x = torch.tensor(prefix_ids, device=device)
    return net.generate(None, x, temperature=temp, top_k=1, top_p=0.0, cfg_scale=0.0)


def build_prefix(item, src_tok, P, PAD):
    ids = src_tok(item)["input_ids"] if isinstance(item, str) else list(item)
    return (ids[:P] + [PAD] * P)[:P]


def load_docs(n):
    items = pickle.load(open(os.path.join(DD, "passages.pickle"), "rb"))[:n]
    return items
