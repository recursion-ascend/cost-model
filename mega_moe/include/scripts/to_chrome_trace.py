import os
#!/usr/bin/env python3
"""Convert prof_paired.csv to Chrome Trace Event JSON for MindStudio Insight.

Supports MIX_1_1, MIX_1_2, and AIV-only core layouts (auto-detected from CSV).
Flow arrows are configured via --flow-map JSON file.

Single MegaMoE process group per rank file, one thread per physical execution lane.
MIX_1_2 order: AIC0-01, AIV0-01, AIV1-01, AIC0-02, ...
"""

import pandas as pd
import json
import sys
import os
import argparse


def detect_layout(df, aic_cores_override):
    """Auto-detect core layout from CSV data for a single rank.

    Returns (mode, aic_cores):
      mode      - 'MIX_1_1', 'MIX_1_2', or 'AIV_ONLY'
      aic_cores - number of AIC cores per rank (used for pid calculation)
    """
    aiv_ids = sorted(df[df['role'] == 'AIV']['local_id'].astype(int).unique())
    aic_ids = sorted(df[df['role'] == 'AIC']['local_id'].astype(int).unique())

    aiv_count = len(aiv_ids)
    aic_count = len(aic_ids)

    if aic_count == 0 and aiv_count > 0:
        mode = 'AIV_ONLY'
        aic_cores = 0
    elif aic_count > 0 and aiv_count == 2 * aic_count:
        mode = 'MIX_1_2'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    elif aic_count > 0 and aiv_count == aic_count:
        mode = 'MIX_1_1'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    elif aic_count > 0:
        # Fallback: assume MIX_1_1 if ratio doesn't match
        mode = 'MIX_1_1'
        aic_cores = aic_cores_override if aic_cores_override > 0 else aic_count
    else:
        mode = 'AIV_ONLY'
        aic_cores = 0

    return mode, aic_cores


def compute_pid_tid(rank, role, local_id, mode, aic_cores):
    """Use flat, uniquely ordered lanes within each rank."""
    lid = int(local_id)
    if mode == 'MIX_1_2':
        tid = lid * 3 if role == 'AIC' else (lid // 2) * 3 + 1 + lid % 2
    elif mode == 'MIX_1_1':
        tid = lid * 2 + (role == 'AIV')
    else:
        tid = lid
    return int(rank), int(tid)


def track_name(tid, mode):
    if mode == 'MIX_1_2':
        kind = ('AIC0', 'AIV0', 'AIV1')[tid % 3]
        return f'{kind}-{tid // 3:02d}'
    if mode == 'MIX_1_1':
        return f'{("AIC", "AIV")[tid % 2]}-{tid // 2:02d}'
    return f'AIV-{tid:02d}'


def build_flow_arrows(events, flow_map):
    """Generate flow arrow events from a flow-map configuration.

    Returns (flow_events list, per-entry arrow counts list).
    """
    flow_events = []
    flow_id = 1
    counts = []

    for fm_entry in flow_map:
        sf_pattern = fm_entry['setflag_pattern']
        wf_pattern = fm_entry['waitflag_pattern']
        payload_factor = fm_entry.get('payload_factor', 0)
        flow_name = fm_entry.get('name', 'SetFlag->WaitFlag')

        # Filter by role (cat field) for generality across layout modes
        setflag_evts = [e for e in events if sf_pattern in e['name'] and e['cat'] == 'AIV']
        waitflag_evts = [e for e in events if wf_pattern in e['name'] and e['cat'] == 'AIC']

        sf_by_payload = {}
        for sf in setflag_evts:
            p = sf['args']['payload']
            sf_by_payload.setdefault(p, []).append(sf)

        wf_by_payload = {}
        for wf in waitflag_evts:
            p = wf['args']['payload']
            wf_by_payload.setdefault(p, []).append(wf)

        # Auto-detect payload factor from waitflag payload stride if not specified
        if payload_factor == 0:
            wf_payloads_sorted = sorted(wf_by_payload.keys())
            if len(wf_payloads_sorted) >= 2:
                payload_factor = wf_payloads_sorted[1] - wf_payloads_sorted[0]
            else:
                payload_factor = 2

        arrow_count = 0
        for sf_payload, sf_list in sf_by_payload.items():
            wf_payload = sf_payload * payload_factor
            wf_list = wf_by_payload.get(wf_payload, [])
            if not wf_list:
                continue

            wf_by_pid = {}
            for wf in wf_list:
                wf_by_pid.setdefault(wf['pid'], []).append(wf)

            for sf in sf_list:
                wf_candidates = wf_by_pid.get(sf['pid'], [])
                if not wf_candidates:
                    continue
                wf = min(wf_candidates,
                         key=lambda w: abs((w['ts'] + w['dur']) - (sf['ts'] + sf['dur'])))
                s_ts = sf['ts'] + sf['dur']
                f_ts = max(wf['ts'] + wf['dur'], s_ts + 0.001)
                flow_events.append({
                    "name": flow_name,
                    "cat": "flow",
                    "ph": "s",
                    "id": flow_id,
                    "pid": sf['pid'],
                    "tid": sf['tid'],
                    "ts": round(s_ts, 3),
                })
                flow_events.append({
                    "name": flow_name,
                    "cat": "flow",
                    "ph": "f",
                    "bp": "e",
                    "id": flow_id,
                    "pid": wf['pid'],
                    "tid": wf['tid'],
                    "ts": round(f_ts, 3),
                })
                flow_id += 1
                arrow_count += 1

        counts.append((flow_name, sf_pattern, wf_pattern, payload_factor, arrow_count))

    return flow_events, counts


def convert(csv_path, out_path, freq, aic_cores_hint, flow_map, rank=None, device_ids=None):
    df = pd.read_csv(csv_path)
    if rank is not None:
        df = df[df['rank'].astype(int) == int(rank)]

    # Detect layout from this rank's data
    mode, aic_cores = detect_layout(df, aic_cores_hint)
    print("[INFO] Rank %s: detected layout=%s, AIC_CORES=%d" % (rank, mode, aic_cores))

    origin = int(df['begin_cycle'].min())
    df = df[~df['event'].isin(['EVT_KERNEL', 'KERNEL'])].copy()
    df['event'] = df['event'].str.removeprefix('EVT_')
    enriched_path = os.path.join(os.path.dirname(csv_path), 'view_enriched.json')
    enriched = {}
    if os.path.exists(enriched_path):
        with open(enriched_path) as f:
            for item in json.load(f):
                enriched[(int(item['rank']),int(item['core_id']),int(item['begin_cycle']),item['event'])] = item
    events = []
    tids_seen = {}       # pid -> set of tid
    tid_roles = {}       # (pid, tid) -> role string

    for _, row in df.iterrows():
        rk = int(row['rank'])
        role = row['role']
        lid = int(row['local_id'])
        pid, tid = compute_pid_tid(rk, role, lid, mode, aic_cores)

        tids_seen.setdefault(pid, set()).add(tid)
        tid_roles[(pid, tid)] = role

        begin_us = (int(row['begin_cycle']) - origin) / freq
        dur_us = float(row['duration_cycles']) / freq

        payload_val = int(row['payload']) if pd.notna(row['payload']) else 0
        display_name = row["event"]
        extra_args = {}
        # payload 位段见 docs/stages.md：GMM 类 [23:16]=全局wave序号 [30:24]=专家号；
        # DISPATCH_SCHEDULE [23:16]=dispatch wave 序号 [15:0]=起始专家号
        if row["event"] in ("GMM1", "ACT_QUANT", "GMM2", "COMBINE",
                            "WAIT_GMM1_INPUT", "WAIT_GMM1_BUFFER", "WAIT_ACT_INPUT",
                            "WAIT_GMM2_INPUT", "WAIT_COMBINE_INPUT"):
            wave = (payload_val >> 16) & 0xff
            expert = (payload_val >> 24) & 0x7f
            display_name = f"{display_name}·w{wave}"
            extra_args = {"wave": wave, "expert": expert}
        elif row["event"] == "DISPATCH_SCHEDULE":
            wave = (payload_val >> 16) & 0xff
            display_name = f"{display_name}·w{wave}"
            extra_args = {"wave": wave, "dispatch_expert": payload_val & 0xffff}

        events.append({
            "name": ("SHARED_" if payload_val & 0x80000000 and row["event"] in ("GMM1", "ACT_QUANT", "GMM2") else "") + display_name,
            "cat": role,
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "ts": round(begin_us, 3),
            "dur": round(dur_us, 3),
            "args": {
                "rank": rk,
                "local_id": lid,
                "payload": payload_val,
                "cycles": int(row['duration_cycles']),
                **extra_args,
                **enriched.get((rk,int(row['core_id']),int(row['begin_cycle']),row['event']), {}),
            }
        })

    # One origin for all cores in this rank; preserve inter-core offsets.

    # Two sibling process groups per rank, sharing timestamps and lane IDs.
    original_events = events
    events = []
    meta = []
    for rank_pid in sorted(tids_seen):
        for view, label in ((0, "MegaMoE · 完整流水"), (1, "MegaMoE · 隐藏 WAIT")):
            pid = rank_pid * 2 + view
            meta.append({"name": "process_name", "ph": "M", "pid": pid,
                         "args": {"name": f"Rank {rank_pid} · Device {device_ids[rank_pid] if device_ids is not None else rank_pid} · {label}"}})
            meta.append({"name": "process_sort_index", "ph": "M", "pid": pid,
                         "args": {"sort_index": pid}})
            for tid in sorted(tids_seen[rank_pid]):
                meta.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                             "args": {"name": track_name(tid, mode)}})
                meta.append({"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid,
                             "args": {"sort_index": tid}})
            for event in original_events:
                if event['pid'] != rank_pid:
                    continue
                if view == 1 and (event['name'].startswith('WAIT_') or
                                  event['args'].get('category') == 'wait'):
                    continue
                events.append(dict(event, pid=pid))

    # Flow arrows from flow-map configuration
    flow_events = []
    flow_info = []
    if flow_map:
        flow_events, flow_info = build_flow_arrows(events, flow_map)
        for name, sf_pat, wf_pat, factor, count in flow_info:
            print("[INFO]   Flow '%s': %d arrows (%s -> %s, factor=%d)" %
                  (name, count, sf_pat, wf_pat, factor))

    all_events = meta + sorted(
        events, key=lambda e: (e['pid'], e['tid'], e['ts'])) + flow_events

    trace = {
        "traceEvents": all_events,
        "displayTimeUnit": "us",
    }

    with open(out_path, 'w') as f:
        json.dump(trace, f, indent=2)
    print("[INFO] %s: %d events, %d groups, %d flow arrows" % (
        out_path, len(events), 2 * len(tids_seen), len(flow_events) // 2))


def main():
    parser = argparse.ArgumentParser(
        description='Convert prof_paired.csv to Chrome Trace Event JSON for MindStudio Insight.')
    parser.add_argument('csv', help='Path to prof_paired.csv')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory (default: same directory as CSV)')
    parser.add_argument('--freq', type=float, default=1600.0,
                        help='NPU clock frequency in MHz for cycle->us conversion (default: 1600.0)')
    parser.add_argument('--aic-cores', type=int, default=0,
                        help='Number of AIC cores per rank for pid calculation (default: 0 = auto-detect)')
    parser.add_argument('--flow-map', type=str, default=None,
                        help='JSON file defining flow arrow relationships (default: no flow arrows)')

    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.output_dir if args.output_dir else os.path.dirname(csv_path) or '.'

    flow_map = None
    if args.flow_map:
        with open(args.flow_map, 'r') as f:
            flow_map = json.load(f)
        print("[INFO] Loaded flow-map: %d entries from %s" % (len(flow_map), args.flow_map))

    ranks = sorted(set(pd.read_csv(csv_path)['rank'].astype(str).unique()))
    print("[INFO] Processing %d rank(s) from %s" % (len(ranks), csv_path))
    from config_io import load_config
    config_path = os.path.abspath(os.path.join(os.path.dirname(csv_path), '..', 'config.json5'))
    device_ids = load_config(config_path)['device']['ids'] if os.path.exists(config_path) else None
    combined = []
    origins = {}
    source = pd.read_csv(csv_path)
    for rk in ranks:
        out_path = os.path.join(out_dir, os.path.basename(os.path.dirname(config_path)) + '_trace_rank%s.json' % rk)
        convert(csv_path, out_path, args.freq, args.aic_cores, flow_map, rank=rk, device_ids=device_ids)
        origins[rk] = int(source[source['rank'].astype(str) == rk]['begin_cycle'].min())
        with open(out_path) as f:
            for event in json.load(f)['traceEvents']:
                if event.get('ph') in ('s', 't', 'f'):
                    event['id'] = f"rank{rk}:{event['id']}"
                combined.append(event)
    merged_path = os.path.join(out_dir, os.path.basename(os.path.dirname(config_path)) + '_trace_all_ranks.json')
    with open(merged_path, 'w') as f:
        json.dump({'traceEvents': combined, 'displayTimeUnit': 'us',
                   'metadata': {'time_alignment': 'per-rank kernel origin; NOT cross-device synchronized',
                                'rank_origin_cycles': origins, 'device_ids': device_ids}}, f, indent=2)
    print('[INFO] Combined ranks: ' + merged_path)


if __name__ == '__main__':
    main()
