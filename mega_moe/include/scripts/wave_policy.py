"""MegaMoE arch35 host wave-sizing policy model (mirror of the C++ host formula).

Single source of truth in Python for:
  - regression checks (old vs new formula)
  - dry-run sweep tables (expected G1/G2/mGroupsPerWave per p1/p2)
  - post-run verification against the host log line

Mirrors, and must stay in sync with:
  - mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp (CalcMGroupsPerWave)
  - megamoe_profile/include/tiling_helpers.h (harness copy)

Only host wave-sizing policy is modeled; the device kernel consumes the final
mGroupsPerWave only, so p1/p2 never appear in the tiling data ABI.
"""
import math

GMM_TILE_N = 256

# p1 tiers (bs-adaptive; default tier keeps historical fixed-4 behavior)
GMM1_MIN_LOGICAL_TILES_PER_CORE = 4
GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL = 2
GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE = 6
GMM1_SMALL_BATCH_TOKEN_THRESHOLD = 2048
GMM1_LARGE_BATCH_TOKEN_THRESHOLD = 16384
# p2: GMM2 per-core min logical N tiles (historically implicit 1)
GMM2_MIN_LOGICAL_TILES_PER_CORE = 1


def ceil_div(a, b):
    return -(-a // b)


def resolve_p1(bs):
    """Default p1 resolution: bs-tier selection (auto policy)."""
    if 0 < bs < GMM1_SMALL_BATCH_TOKEN_THRESHOLD:
        return GMM1_MIN_LOGICAL_TILES_PER_CORE_SMALL
    if bs >= GMM1_LARGE_BATCH_TOKEN_THRESHOLD:
        return GMM1_MIN_LOGICAL_TILES_PER_CORE_LARGE
    return GMM1_MIN_LOGICAL_TILES_PER_CORE


def wave_policy(aic_num, hidden_dim, h, p1, p2):
    """Generalized formula: mGroupsPerWave = max(G1, G2).

    G1 = ceil(aicNum * p1 / ceil(hiddenDim / GMM_TILE_N))
    G2 = ceil(aicNum * p2 / ceil(h / GMM_TILE_N))
    Degenerate inputs (hidden_dim/h/aic_num == 0) short-circuit to 1 like the C++.
    """
    if hidden_dim == 0 or h == 0 or aic_num == 0:
        return {"t1": 0, "t2": 0, "g1": 0, "g2": 0, "mgw": 1, "dominant": "degenerate"}
    t1 = ceil_div(hidden_dim, GMM_TILE_N)
    t2 = ceil_div(h, GMM_TILE_N)
    g1 = ceil_div(aic_num * p1, t1)
    g2 = ceil_div(aic_num * p2, t2)
    dominant = "GMM1" if g1 > g2 else ("GMM2" if g2 > g1 else "crossover")
    return {"t1": t1, "t2": t2, "g1": g1, "g2": g2, "mgw": max(g1, g2), "dominant": dominant}


def old_wave_policy(aic_num, hidden_dim, h, bs):
    """Reference: HEAD@ebf6f3a formula (tiered p1, implicit p2=1)."""
    return wave_policy(aic_num, hidden_dim, h, resolve_p1(bs), GMM2_MIN_LOGICAL_TILES_PER_CORE)


def baseline_wave_policy(aic_num, hidden_dim, h):
    """Reference: imported-baseline b258e58 formula (fixed p1=4, implicit p2=1)."""
    return wave_policy(aic_num, hidden_dim, h, 4, GMM2_MIN_LOGICAL_TILES_PER_CORE)


def expected_policy(aic_num, hidden_dim, h, bs, p1_override=0, p2_override=0):
    """Effective policy for a run: override (>0) replaces tier resolution."""
    p1 = p1_override if p1_override > 0 else resolve_p1(bs)
    p2 = p2_override if p2_override > 0 else GMM2_MIN_LOGICAL_TILES_PER_CORE
    result = wave_policy(aic_num, hidden_dim, h, p1, p2)
    result["p1"] = p1
    result["p2"] = p2
    return result
