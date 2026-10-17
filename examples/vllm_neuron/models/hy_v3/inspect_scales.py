import json, sys, torch
from safetensors import safe_open

root = sys.argv[1]
wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
handles = {}


def get(k):
    f = wm[k]
    if f not in handles:
        handles[f] = safe_open(f"{root}/{f}", "pt")
    return handles[f].get_tensor(k)


print(
    "keys with kv/k_scale:", [k for k in wm if k.endswith(("k_scale", "v_scale"))][:4]
)
bad_qkv = 0
for L in range(81):
    p = f"model.layers.{L}.self_attn"
    s = [get(f"{p}.{n}_proj.input_scale").float().item() for n in "qkv"]
    if not (s[0] == s[1] == s[2]):
        bad_qkv += 1
    if L in (0, 1, 40, 79, 80):
        print(
            L,
            "qkv in",
            s,
            "o in",
            get(f"{p}.o_proj.input_scale").item(),
            "w",
            [get(f"{p}.{n}_proj.weight_scale").item() for n in "qkvo"],
        )
print("layers with unequal qkv input_scale:", bad_qkv)
for L in (1, 40, 79):
    p = f"model.layers.{L}.mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        ws = torch.tensor(
            [
                get(f"{p}.experts.{e}.{proj}.weight_scale").float().item()
                for e in range(192)
            ]
        )
        ins = torch.tensor(
            [
                get(f"{p}.experts.{e}.{proj}.input_scale").float().item()
                for e in range(192)
            ]
        )
        print(
            L,
            proj,
            "w_scale min/max %.3g/%.3g" % (ws.min(), ws.max()),
            "in_scale min/max %.3g/%.3g uniq %d"
            % (ins.min(), ins.max(), ins.unique().numel()),
        )
    g = torch.tensor(
        [
            get(f"{p}.experts.{e}.gate_proj.input_scale").float().item()
            for e in range(192)
        ]
    )
    u = torch.tensor(
        [get(f"{p}.experts.{e}.up_proj.input_scale").float().item() for e in range(192)]
    )
    print("  gate==up input scale:", bool((g == u).all()))
    sm = [
        get(f"{p}.shared_mlp.{n}.{t}").float().item()
        for n in ("gate_proj", "up_proj", "down_proj")
        for t in ("weight_scale", "input_scale")
    ]
    print(
        "  shared",
        sm,
        "expert_bias",
        get(f"{p}.expert_bias").dtype,
        get(f"{p}.expert_bias")[:4],
        "router",
        get(f"{p}.router.gate.weight").dtype,
    )
w = get("model.layers.1.mlp.experts.0.gate_proj.weight").float()
print(
    "fp8 weight absmax",
    w.abs().max().item(),
    "frac>240",
    (w.abs() > 240).float().mean().item(),
)
w = get("model.layers.0.self_attn.q_proj.weight").float()
print(
    "q weight absmax",
    w.abs().max().item(),
    "frac>240",
    (w.abs() > 240).float().mean().item(),
)
for k in (
    "model.layers.80.enorm.weight",
    "model.layers.80.eh_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
):
    t = get(k)
    print(k, t.dtype, tuple(t.shape))
