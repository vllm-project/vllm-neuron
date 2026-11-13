"""Merge consecutive optimized fused decode launches into one (round 5: launch entries fewer a step).

    python merge_launch.py <fused dir> <merged plan.json (the model's, dumped by DEEPSEEK_V41_FUSED_PLAN_DUMP)>
                           <launch 0> <launch 1> [<launch 2> ...] [--no-handoff]

(layers_decode_L08_L23_tp64: the L08_L15 and L16_L23 launches; layers_decode_L24_L39_head_tp64: the L24_L31 and
L32_L39_head launches with --no-handoff -- with the hand-off that kernel compiled slower.)

Writes <fused dir>/<merged name>/: `l<k>/` = launch k's package (its member packages, moe_tp, fused_member:
their relative imports stay inside; the parts are 8-block launches, not merged ones), each part's
`fused_member.py` with
a member-index offset (scratch names `L<offset + m>_...` unique in the merged kernel), `plan.json` = the
merged plan, and `<merged name>.py`: the merged entry, which calls the parts' bodies in order (a part's
arguments: the merged parameters, or an earlier part's returned outputs -- the stream / pre-mix, a top-k)
and returns the merged plan's outputs (the parts' returns)."""
import ast
import json
import shutil
import sys
from pathlib import Path

argv = [a for a in sys.argv[1:] if a != "--no-handoff"]
#: `--no-handoff`: no b1 hand-off across the parts' boundaries (the parts' first blocks load their own weights)
HANDOFF = "--no-handoff" not in sys.argv[1:]
fused, planf, parts = Path(argv[0]), Path(argv[1]), argv[2:]
merged = json.loads(planf.read_text())
name = merged["launch"]
plans = [json.loads((fused / p / "plan.json").read_text()) for p in parts]
offs = [0]
for pl in plans:
    offs.append(offs[-1] + len(pl["members"]))
assert len(merged["members"]) == offs[-1], ("member counts", len(merged["members"]), offs)


def refs(args):
    return args.split()


def mapping(pl, off):
    """part param j -> the merged plan's ref (a parameter or an earlier member's output)"""
    out = {}
    for m, mem in enumerate(pl["members"]):
        mm = merged["members"][off + m]
        assert mm["kernel"].split(":")[-1] == mem["kernel"].split(":")[-1], (off + m, mm["kernel"], mem["kernel"])
        for r_l, r_m in zip(refs(mem["args"]), refs(mm["args"])):
            if r_l.startswith("p"):
                j = int(r_l[1:])
                prev = out.get(j)
                assert prev is None or prev == r_m, (off + m, r_l, prev, r_m)
                out[j] = r_m
            else:
                mo, k = r_l[1:].split(".")
                want = f"o{int(mo) + off}.{k}"
                assert r_m == want, (off + m, r_l, r_m, want)
    assert sorted(out) == list(range(pl["nparams"])), "unmapped params"
    return out


maps = [mapping(pl, off) for pl, off in zip(plans, offs)]
# each part's returns as merged member outputs: ref -> (part, index)
rets = {}
for k, (pl, off) in enumerate(zip(plans, offs)):
    for i, r in enumerate(refs(pl["returns"])):
        mo, kk = r[1:].split(".")
        rets[f"o{int(mo) + off}.{kk}"] = (k, i)


def expr(r, before=None):
    if r.startswith("p"):
        return f"p{int(r[1:])}"
    if r in rets and (before is None or rets[r][0] < before):
        k, i = rets[r]
        return f"_r{k}[{i}]"
    raise SystemExit(f"merged ref {r} is neither a parameter nor an earlier part's returned output")


out = fused / name
if out.exists():
    shutil.rmtree(out)
out.mkdir()
for k, p in enumerate(parts):
    sub = out / f"l{k}"
    shutil.copytree(fused / p, sub, ignore=shutil.ignore_patterns("__pycache__"))
    (sub / "plan.json").unlink()    # (not a launch of its own: FUSED_LAUNCHES lists dirs with plan.json)
    for fm in sub.rglob("fused_member.py"):
        s = fm.read_text()
        if "index + OFFSET" not in s:
            s = s.replace('''_CACHE_COPY_FLAGS = ("CACHE_COPY_EARLY", "CACHE_COPY_AT_AR")
''', '''_CACHE_COPY_FLAGS = ("CACHE_COPY_EARLY", "CACHE_COPY_AT_AR")
#: the members' index offset in a merged launch (this launch's members after others')
OFFSET = 0
''')
            s = s.replace('''name=f"L{index}_{kwargs['name']}")''', '''name=f"L{index + OFFSET}_{kwargs['name']}")''')
            s = s.replace('''name=f"L{index}_out{anon[0]}")''', '''name=f"L{index + OFFSET}_out{anon[0]}")''')
            assert "index + OFFSET" in s, fm
            fm.write_text(s)
# share-dict launches (`share["b1"] = {...}` in the body): export the dict, inherit the previous part's
# launch-level tiles (persistent buffers, constants, tables; not the per-pair hand-off state) -- else the
# earlier part's persistent tiles stay allocated next to this part's and the blocks spill
inherit = []
for k, p in enumerate(parts):
    lf = out / f"l{k}" / f"{p}.py"
    s2 = lf.read_text()
    i = s2.find('    share["b1"] = {')
    if i < 0:
        inherit.append(False)
        continue
    j = s2.index("}}\n", i) + 3
    s2 = s2[:j] + """    LAST_SHARE[0] = share
    if INHERIT[0] is not None:
        # (merged launch: the previous part's launch-level SBUF tiles and tables)
        for _k, _v in INHERIT[0].items():
            if _k not in _HANDOFF:
                share["b1"].setdefault(_k, _v)
""" + s2[j:]
    d = s2.index("@nki.jit")
    s2 = s2[:d] + """#: merged launches (merge_launch.py): this launch's b1 share dict, and the previous part's to inherit
LAST_SHARE = [None]
INHERIT = [None]
_HANDOFF = ("attn_next", "next_b1", "pre_next", "S", "xin", "S_src", "gate", "pre", "pref")


""" + s2[d:]
    lf.write_text(s2)
    inherit.append(True)
# the b1 hand-off across a part boundary: the earlier part's b1 dict gets the pair (its last layer -> the next
# part's first: the weight / constant prefetch, the stream left in SBUF) and the next part adopts that dict
def b1_dict_expr(src):
    if '    share["b1"] = {' in src:
        return 'share["b1"]'
    if '    r2_decode.B1_SHARE = {"b1": {' in src:
        return 'r2_decode.B1_SHARE["b1"]'
    return None


def param_names(lf, fname):
    for n in ast.walk(ast.parse(lf.read_text())):
        if isinstance(n, ast.FunctionDef) and n.name == fname:
            return [a.arg for a in n.args.args]
    raise SystemExit(f"no function {fname} in {lf}")


names = [param_names(out / f"l{k}" / f"{p}.py", p) for k, p in enumerate(parts)]
layer_of = lambda nm: int(nm[1:3]) if nm[:1] == "L" and nm[1:3].isdigit() else None
handoff = []
for k, p in enumerate(parts):
    lf = out / f"l{k}" / f"{p}.py"
    s2 = lf.read_text()
    d_expr = b1_dict_expr(s2) if HANDOFF else None
    handoff.append((d_expr,))
    if d_expr is None:
        continue
    m0 = s2.index("    with member(0,")
    code = f"""    # (merged launch, merge_launch.py: the b1 hand-off across this part's boundaries)
    _d_own = {d_expr}
    if ADOPT[0] is not None:
        _d = ADOPT[0]
        _d["attn_next"].update(_d_own["attn_next"])
        _d["next_b1"] |= _d_own["next_b1"]
        _d["pre_next"].update(_d_own["pre_next"])
        {d_expr} = _d
    if EXTRA_NEXT[0] is not None:
        _last, _aw_n, _pre_n = EXTRA_NEXT[0]
        _key = id(_loc[f"L{{_last:02d}}_wq_a_s"]) if f"L{{_last:02d}}_wq_a_s" in _loc else id(_loc[f"L{{_last}}_wq_a_s"])
        {d_expr}["attn_next"][_key] = _aw_n
        {d_expr}["next_b1"].add(_key)
        {d_expr}["pre_next"][_key] = _pre_n
    B1_DICT[0] = {d_expr}
"""
    s2 = s2[:m0] + code + s2[m0:]
    d = s2.index("@nki.jit")
    s2 = s2[:d] + """#: merged launches (merge_launch.py): the previous part's b1 dict to adopt, the next part's first layer to hand
#: off to (last layer, its attention tensors, its pre-mix constants), this part's b1 dict
ADOPT = [None]
EXTRA_NEXT = [None]
B1_DICT = [None]


""" + s2[d:]
    lf.write_text(s2)
(out / "plan.json").write_text(json.dumps(merged, indent=1) + "\n")
(out / "__init__.py").write_text("")
params = ", ".join(f"p{j}" for j in range(merged["nparams"]))
lines = [
    "# SPDX-License-Identifier: Apache-2.0",
    f'"""The merged decode launch {name} (round 5): {", ".join(parts)} in one kernel (launch entry',
    "barriers, preambles and turnarounds fewer a step). Generated by merge_launch.py: `l<k>/` are the parts'",
    'packages; a part takes an earlier part\'s returned outputs (the stream / pre-mix, a top-k)."""',
    "import nki",
    "",
]
for k, p in enumerate(parts):
    lines.append(f"from .l{k} import {p} as _p{k}")
    lines.append(f"from .l{k} import fused_member as _fm{k}")
lines += ["", "", "@nki.jit", f"def {name}({params}):"]
for k, (p, pl, mp, off) in enumerate(zip(parts, plans, maps, offs)):
    args = ", ".join(expr(mp[j], before=k) for j in range(pl["nparams"]))
    nxt = k + 1 < len(parts) and handoff[k][0] is not None and handoff[k + 1][0] is not None
    if nxt:
        nm = names[k + 1]
        first = layer_of(nm[0])
        last = max(layer_of(n) for n in names[k] if layer_of(n) is not None and n.endswith("_wq_a_s"))
        def ref(t):
            for cand in (f"L{first:02d}_{t}", f"L{first}_{t}"):
                if cand in nm:
                    return expr(maps[k + 1][nm.index(cand)], before=k)
            raise SystemExit(f"no {t} of layer {first} in {parts[k + 1]}")
        aw = ", ".join(ref(t) for t in ("wq_a", "wq_a_s", "wq_b", "wq_b_s", "wkv", "wkv_s", "wo_a", "wo_a_s", "wo_b", "wo_b_s"))
        pre = ", ".join(ref(t) for t in ("attn_norm_w", "hca_fn_t", "hca_base", "hca_scale"))
        lines.append(f"    _p{k}.EXTRA_NEXT[0] = ({last}, ({aw}), ({pre}))")
    if k > 0 and handoff[k][0] is not None and handoff[k - 1][0] is not None:
        lines.append(f"    _p{k}.ADOPT[0] = _p{k - 1}.B1_DICT[0]")
    elif k > 0 and inherit[k] and inherit[k - 1]:
        lines.append(f'    _p{k}.INHERIT[0] = _p{k - 1}.LAST_SHARE[0].get("b1")')
    lines.append(f"    _fm{k}.OFFSET = {off}")
    lines.append(f"    _r{k} = _p{k}.{p}.func({args})")
    lines.append(f"    _fm{k}.OFFSET = 0")
    if nxt:
        lines.append(f"    _p{k}.EXTRA_NEXT[0] = None")
    if k > 0 and handoff[k][0] is not None and handoff[k - 1][0] is not None:
        lines.append(f"    _p{k}.ADOPT[0] = None")
lines.append(f"    return {', '.join(expr(r) for r in refs(merged['returns']))}")
(out / f"{name}.py").write_text("\n".join(lines) + "\n")
print("merged", name, "params", merged["nparams"], "parts", [pl["nparams"] for pl in plans], "->", out)
