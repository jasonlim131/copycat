"""CLI entry point for the unlearning + interpretability experiment.

Subcommands:
    prepare-data     : (re)build forget/retain/trivia JSONL files
    baseline-eval    : evaluate the untouched base model
    unlearn          : run RMU or NPO and save the unlearned model
    eval             : evaluate base or an unlearned checkpoint
    interp           : run all four interpretability analyses
    report           : assemble a markdown summary table
    run-all          : do all of the above end-to-end
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.cfg import Config, load_config, seed_everything
from src.data_utils import Fact, read_jsonl
from src.eval_harness import exact_match_acc, trivia_acc, wikitext_ppl
from src.interp import (
    hidden_cosine,
    linear_probe_layerwise,
    logit_lens_acc,
    topk_logit_drop,
)
from src.model_utils import assert_pythia_410m, load_model_tokenizer
from src.plotting import plot_layerwise, plot_logit_drop_bars

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "config.yaml"


# ---------------------------------------------------------------------------

def _load_cfg(args) -> Config:
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    return cfg


def _load_facts(cfg: Config) -> Dict[str, List[Fact]]:
    return {
        "forget": read_jsonl(cfg.abspath(cfg.paths.forget)),
        "retain": read_jsonl(cfg.abspath(cfg.paths.retain)),
        "trivia": read_jsonl(cfg.abspath(cfg.paths.trivia)),
    }


def _load_model(cfg: Config, ckpt: str | None):
    if ckpt is None or ckpt == "base":
        model, tok = load_model_tokenizer(cfg.model_name, cfg.torch_dtype(), cfg.device)
    else:
        ckpt_path = cfg.abspath(cfg.paths.checkpoints) / ckpt
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, torch_dtype=cfg.torch_dtype()
        ).to(cfg.device)
    assert_pythia_410m(model)
    return model, tok


def _save_metrics(cfg: Config, name: str, metrics: dict) -> Path:
    out = cfg.abspath(cfg.paths.metrics) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    return out


# --- subcommands -----------------------------------------------------------

def cmd_prepare_data(args) -> None:
    from data import build_data  # type: ignore[import-not-found]
    build_data.main()


def cmd_baseline_eval(args) -> None:
    cfg = _load_cfg(args)
    facts = _load_facts(cfg)
    model, tok = _load_model(cfg, "base")
    metrics = _eval_all(cfg, model, tok, facts)
    p = _save_metrics(cfg, "base", metrics)
    print(f"saved {p}")


def cmd_unlearn(args) -> None:
    cfg = _load_cfg(args)
    facts = _load_facts(cfg)
    model, tok = _load_model(cfg, "base")

    if args.method == "rmu":
        from src.rmu import save, train_rmu
        stats = train_rmu(cfg, model, tok, facts["forget"], facts["retain"])
        out = save(model, cfg, "rmu")
    elif args.method == "npo":
        from src.npo import train_npo
        stats = train_npo(cfg, model, tok, facts["forget"], facts["retain"])
        out = cfg.abspath(cfg.paths.checkpoints) / "npo"
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
    else:
        raise ValueError(args.method)

    _save_metrics(cfg, f"{args.method}_train_stats", stats)
    print(f"saved unlearned model to {out}")


def cmd_eval(args) -> None:
    cfg = _load_cfg(args)
    facts = _load_facts(cfg)
    model, tok = _load_model(cfg, args.method)
    metrics = _eval_all(cfg, model, tok, facts)
    p = _save_metrics(cfg, args.method, metrics)
    print(f"saved {p}")


def _eval_all(cfg: Config, model, tok, facts) -> dict:
    forget = exact_match_acc(model, tok, facts["forget"], cfg.device,
                             cfg.eval.max_new_tokens_pad, use_paraphrases=False)
    forget_para = exact_match_acc(model, tok, facts["forget"], cfg.device,
                                  cfg.eval.max_new_tokens_pad, use_paraphrases=True)
    retain = exact_match_acc(model, tok, facts["retain"], cfg.device,
                             cfg.eval.max_new_tokens_pad, use_paraphrases=False)
    trivia = trivia_acc(model, tok, facts["trivia"], cfg.device, cfg.eval.max_new_tokens_pad)
    ppl = wikitext_ppl(model, tok, cfg.device, cfg.eval.wikitext_split, cfg.eval.wikitext_stride)
    return {
        "forget_em": forget["accuracy"],
        "forget_paraphrase_em": forget_para["accuracy"],
        "retain_em": retain["accuracy"],
        "trivia_em": trivia["accuracy"],
        "wikitext_ppl": ppl,
    }


def cmd_interp(args) -> None:
    cfg = _load_cfg(args)
    facts = _load_facts(cfg)
    base_model, tok = _load_model(cfg, "base")
    other_model, _ = _load_model(cfg, args.method)

    layers = list(cfg.interp.layers)
    fig_dir = cfg.abspath(cfg.paths.figures)

    # 1) Logit-lens accuracy on forget vs retain, base vs unlearned.
    ll_base_f = logit_lens_acc(base_model, tok, facts["forget"], layers, cfg.device, cfg.train.max_seq_len)
    ll_base_r = logit_lens_acc(base_model, tok, facts["retain"], layers, cfg.device, cfg.train.max_seq_len)
    ll_oth_f = logit_lens_acc(other_model, tok, facts["forget"], layers, cfg.device, cfg.train.max_seq_len)
    ll_oth_r = logit_lens_acc(other_model, tok, facts["retain"], layers, cfg.device, cfg.train.max_seq_len)
    plot_layerwise(
        {f"base/forget": ll_base_f, f"base/retain": ll_base_r,
         f"{args.method}/forget": ll_oth_f, f"{args.method}/retain": ll_oth_r},
        title=f"Logit-lens accuracy ({args.method} vs base)",
        ylabel="acc",
        out_path=fig_dir / f"logit_lens_{args.method}.png",
    )

    # 2) Hidden-state cosine on forget prompts.
    cos_f = hidden_cosine(base_model, other_model, tok, facts["forget"], layers, cfg.device, cfg.train.max_seq_len)
    cos_r = hidden_cosine(base_model, other_model, tok, facts["retain"], layers, cfg.device, cfg.train.max_seq_len)
    plot_layerwise(
        {"forget": cos_f, "retain": cos_r},
        title=f"Hidden-state cosine: base vs {args.method}",
        ylabel="cos(h_base, h_unlearned)",
        out_path=fig_dir / f"hidden_cosine_{args.method}.png",
    )

    # 3) Top-k logit drop (final layer, gold token).
    rows = topk_logit_drop(base_model, other_model, tok, facts["forget"], cfg.device, cfg.train.max_seq_len)
    plot_logit_drop_bars(rows, title=f"Final-layer gold-token logit drop ({args.method})",
                         out_path=fig_dir / f"logit_drop_{args.method}.png")

    # 4) Linear probe at each layer (does the *base* model linearly separate forget vs retain?).
    probe_acc = linear_probe_layerwise(
        base_model, tok, facts["forget"], facts["retain"],
        layers, cfg.device, cfg.interp.probe_test_frac, cfg.train.max_seq_len, cfg.seed,
    )
    plot_layerwise(
        {"base-model probe": probe_acc},
        title="Linear-probe accuracy: 'is forget fact' (base model)",
        ylabel="probe acc",
        out_path=fig_dir / "linear_probe_base.png",
    )

    interp_dump = {
        "logit_lens": {"base_forget": ll_base_f, "base_retain": ll_base_r,
                       f"{args.method}_forget": ll_oth_f, f"{args.method}_retain": ll_oth_r},
        "hidden_cosine": {"forget": cos_f, "retain": cos_r},
        "logit_drop_rows": rows,
        "probe_acc": probe_acc,
    }
    p = _save_metrics(cfg, f"interp_{args.method}", interp_dump)
    print(f"saved {p}")


def cmd_report(args) -> None:
    cfg = _load_cfg(args)
    metrics_dir = cfg.abspath(cfg.paths.metrics)
    base = json.loads((metrics_dir / "base.json").read_text())
    rows = [("metric", "base", "rmu", "npo")]
    method_files = {m: metrics_dir / f"{m}.json" for m in ("rmu", "npo")}
    others = {m: (json.loads(p.read_text()) if p.exists() else {}) for m, p in method_files.items()}
    keys = ["forget_em", "forget_paraphrase_em", "retain_em", "trivia_em", "wikitext_ppl"]
    for k in keys:
        rows.append((
            k,
            f"{base.get(k, float('nan')):.4f}",
            f"{others['rmu'].get(k, float('nan')):.4f}",
            f"{others['npo'].get(k, float('nan')):.4f}",
        ))
    md = ["# Unlearning experiment results", ""]
    md.append("| " + " | ".join(rows[0]) + " |")
    md.append("|" + "|".join(["---"] * 4) + "|")
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    out = cfg.abspath(cfg.paths.metrics) / "REPORT.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    print("\n".join(md))


def cmd_run_all(args) -> None:
    cmd_prepare_data(args)
    cmd_baseline_eval(args)
    for method in ("rmu", "npo"):
        sub = argparse.Namespace(**vars(args))
        sub.method = method
        cmd_unlearn(sub)
        cmd_eval(sub)
        cmd_interp(sub)
    cmd_report(args)


# --- entry -----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CFG))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare-data").set_defaults(func=cmd_prepare_data)
    sub.add_parser("baseline-eval").set_defaults(func=cmd_baseline_eval)

    s_un = sub.add_parser("unlearn")
    s_un.add_argument("--method", choices=["rmu", "npo"], required=True)
    s_un.set_defaults(func=cmd_unlearn)

    s_ev = sub.add_parser("eval")
    s_ev.add_argument("--method", choices=["base", "rmu", "npo"], required=True)
    s_ev.set_defaults(func=cmd_eval)

    s_in = sub.add_parser("interp")
    s_in.add_argument("--method", choices=["rmu", "npo"], required=True)
    s_in.set_defaults(func=cmd_interp)

    sub.add_parser("report").set_defaults(func=cmd_report)
    sub.add_parser("run-all").set_defaults(func=cmd_run_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
