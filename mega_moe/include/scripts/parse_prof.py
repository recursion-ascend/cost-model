import os
#!/usr/bin/env python3
"""Parse prof_rank*.bin to prof.csv + prof_paired.csv.

Usage:
    python3 parse_prof.py [options] <prof_rank0.bin> [prof_rank1.bin ...]

Options:
    --n-cores N           Total cores (0=auto-detect from file size)
    --aiv-cores N         AIV core count (0=auto from known configs)
    --pair-map FILE       (Required) JSON file: {"begin_id": end_id, ...} for explicit pairing
    --event-names-from F  Parse profiler.h enum ProfEventId for readable names
"""
import numpy as np, csv, sys, os, argparse, json, re

SLOTS = 4096
REC = 16
ALIGN = 64

KNOWN_CONFIGS = {
    56: (28, "MIX_AIC_1_1 (28 AIV + 28 AIC)"),
    84: (56, "MIX_AIC_1_2 (56 AIV + 28 AIC)"),
    # Note: 950PR AIV-only mode also has 56 cores (56 AIV, 0 AIC).
    # Use --aiv-cores 56 to override auto-detection for AIV-only mode.
}


def detect_n_cores(fpath):
    sz = os.path.getsize(fpath)
    per_core = ALIGN + SLOTS * REC
    if sz < per_core:
        print("[ERROR] File %s too small (%d bytes, need >= %d for at least 1 core)" % (fpath, sz, per_core))
        sys.exit(1)
    n = sz // per_core
    if n * per_core != sz:
        print("[WARN] File size %d not divisible by per-core size %d, guessing N_CORES=%d" % (sz, per_core, n))
    return n


def resolve_aiv_cores(n_cores, user_aiv):
    if user_aiv > 0:
        return user_aiv
    if n_cores in KNOWN_CONFIGS:
        aiv, desc = KNOWN_CONFIGS[n_cores]
        print("[INFO] Auto-detected config: %s" % desc)
        return aiv
    print("[WARN] N_CORES=%d not in known configs. Defaulting aiv_cores=%d. "
          "Use --aiv-cores to override." % (n_cores, n_cores // 2))
    return n_cores // 2


def parse_event_names_from_header(header_path):
    names = {}
    with open(header_path) as f:
        text = f.read()
    for m in re.finditer(r'([A-Z][A-Z0-9_]*(?:_BEGIN|_END))\s*=\s*(0x[0-9a-fA-F]+|\d+)', text):
        name = m.group(1)
        val_str = m.group(2)
        val = int(val_str, 16) if val_str.startswith('0x') else int(val_str)
        names[val] = name
    return names


def parse_one(fpath, n_cores, aiv_cores, event_names):
    raw = open(fpath, 'rb').read()
    ringidx_region = n_cores * ALIGN
    record_bytes = len(raw) - ringidx_region
    if record_bytes <= 0 or record_bytes % (n_cores * REC):
        raise ValueError("Invalid profiler buffer size for the specified core count")
    slots = record_bytes // (n_cores * REC)
    ring_counts = np.frombuffer(raw[:ringidx_region], dtype='<u4')[::(ALIGN // 4)]
    arr = np.frombuffer(raw[ringidx_region:], dtype=np.dtype([
        ('eid', '<u4'), ('payload', '<u4'), ('cycle', '<u8')
    ])).reshape(n_cores, slots)

    overflow_cores = []
    rows = []
    for cid in range(n_cores):
        cnt = int(ring_counts[cid])
        if cnt >= slots:
            overflow_cores.append(cid)
        role = "AIV" if cid < aiv_cores else "AIC"
        lid = cid if cid < aiv_cores else cid - aiv_cores
        for i in range(min(cnt, slots)):
            rec = arr[cid][i]
            eid = int(rec['eid'])
            if eid in event_names:
                name = event_names[eid]
            elif eid == 0x0001:
                name = "KERNEL_BEGIN"
            elif eid == 0x00FF:
                name = "KERNEL_END"
            else:
                name = "EVT_0x%04X" % eid
            rows.append(dict(
                core_id=cid, role=role, local_id=lid,
                event_id=eid,
                event_name=name,
                payload=int(rec['payload']),
                cycle=int(rec['cycle'])))

    if overflow_cores:
        print("[WARN] Ring buffer overflow on %d core(s): %s (events >= %d, data may be truncated)" % (
            len(overflow_cores), overflow_cores[:10], slots))

    return rows


def parse_int(s):
    """Parse integer from string, supporting both decimal and hex (0x prefix)."""
    s = str(s).strip()
    return int(s, 16) if s.startswith('0x') or s.startswith('0X') else int(s)


def build_pair_map(user_pair_map):
    return {parse_int(k): parse_int(v) for k, v in user_pair_map.items()}


def pair_events(rows, pair_map):
    end_to_begin = {e: b for b, e in pair_map.items()}

    open_begin = {}
    pairs = []
    for r in sorted(rows, key=lambda x: x['cycle']):
        eid = r['event_id']
        end_eid = pair_map.get(eid)
        if end_eid:
            open_begin.setdefault((r['core_id'], end_eid), []).append(r)
        elif eid in end_to_begin:
            key = (r['core_id'], eid)
            if key in open_begin and open_begin[key]:
                b = open_begin[key].pop(0)
                begin_name = b['event_name']
                event_name = begin_name.replace('_BEGIN', '') if begin_name.endswith('_BEGIN') else begin_name
                pairs.append(dict(
                    core_id=r['core_id'], role=r['role'], local_id=r['local_id'],
                    event=event_name,
                    begin_cycle=b['cycle'], end_cycle=r['cycle'],
                    duration_cycles=r['cycle'] - b['cycle'],
                    payload=b['payload']))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Parse profiler binary to CSV")
    parser.add_argument('files', nargs='+', help='prof_rank*.bin files')
    parser.add_argument('--n-cores', type=int, default=0, help='Total cores (0=auto-detect)')
    parser.add_argument('--aiv-cores', type=int, default=0, help='AIV core count (0=auto from known configs)')
    parser.add_argument('--pair-map', required=True, help='JSON file with explicit {begin_id: end_id} mapping')
    parser.add_argument('--event-names-from', default=None, help='Path to profiler.h to extract event names')
    args = parser.parse_args()

    event_names = {}
    if args.event_names_from:
        event_names = parse_event_names_from_header(args.event_names_from)
        print("[INFO] Loaded %d event names from %s" % (len(event_names), args.event_names_from))

    n_cores = args.n_cores or detect_n_cores(args.files[0])
    aiv_cores = resolve_aiv_cores(n_cores, args.aiv_cores)
    print("[INFO] N_CORES=%d, AIV_CORES=%d, AIC_CORES=%d" % (n_cores, aiv_cores, n_cores - aiv_cores))

    try:
        with open(args.pair_map) as f:
            user_pair_map = json.load(f)
    except FileNotFoundError:
        print("[ERROR] pair-map file not found: %s" % args.pair_map); sys.exit(1)
    except json.JSONDecodeError as e:
        print("[ERROR] Invalid JSON in %s: %s" % (args.pair_map, e)); sys.exit(1)
    print("[INFO] Loaded explicit pair map: %d pairs" % len(user_pair_map))

    all_rows, all_pairs = [], []
    pair_map = build_pair_map(user_pair_map)
    for fp in args.files:
        rk = os.path.basename(fp).replace('prof_rank', '').replace('.bin', '')
        rows = parse_one(fp, n_cores, aiv_cores, event_names)
        for r in rows:
            r['rank'] = rk
        all_rows.extend(rows)
        pairs = pair_events(rows, pair_map)
        for p in pairs:
            p['rank'] = rk
        all_pairs.extend(pairs)

    for fn, data, fields in [
        ('prof.csv', all_rows, ['rank', 'core_id', 'role', 'local_id',
         'event_id', 'event_name', 'payload', 'cycle']),
        ('prof_paired.csv', all_pairs, ['rank', 'core_id', 'role', 'local_id',
         'event', 'begin_cycle', 'end_cycle', 'duration_cycles', 'payload'])]:
        with open(fn, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print("[INFO] %s: %d rows" % (fn, len(data)))

    print("[INFO] Paired %d event types" % len(pair_map))


if __name__ == '__main__':
    main()
