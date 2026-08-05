#!/usr/bin/env python3
import csv, json, math, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def pct(values, q):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values: return None
    pos = (len(values)-1)*q; lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)


def stat(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return {"avg": round(statistics.mean(values),3) if values else None,
            "p50": round(pct(values,.5),3) if values else None,
            "p95": round(pct(values,.95),3) if values else None,
            "p99": round(pct(values,.99),3) if values else None,
            "max": round(max(values),3) if values else None}


def counter_delta(samples, engine, key):
    vals=[x.get("engines",{}).get(engine,{}).get(key) for x in samples]
    vals=[x for x in vals if x is not None]
    return round(vals[-1]-vals[0],3) if len(vals)>1 else None


def benchmark_samples(benchmark, observations):
    ended = datetime.strptime(benchmark["date"], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    started = ended - timedelta(seconds=float(benchmark["duration"]))
    selected = []
    for record in observations:
        try: when = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        except (KeyError, ValueError): continue
        if started <= when <= ended: selected.append(record)
    return selected


def main():
    root=Path(sys.argv[1]); rows=[]; details=[]
    for mode_dir in sorted(root.glob('D[0-2]')):
        observations=[]
        obs_path=mode_dir/'observations.jsonl'
        if obs_path.exists():
            for line in obs_path.read_text(errors='replace').splitlines():
                try: observations.append(json.loads(line))
                except json.JSONDecodeError: pass
        for path in sorted((mode_dir/'benchmarks').glob('*.json')):
            b=json.loads(path.read_text()); case=path.stem
            samples=benchmark_samples(b, observations)
            row={"mode":mode_dir.name,"case":case,"completed":b.get('completed'),"failed":b.get('failed'),
                 "duration_s":b.get('duration'),"output_tok_s":b.get('output_throughput'),
                 "goodput_req_s":b.get('request_goodput'),"ttft_p50_ms":b.get('p50_ttft_ms'),
                 "ttft_p95_ms":b.get('p95_ttft_ms'),"ttft_p99_ms":b.get('p99_ttft_ms'),
                 "tpot_p50_ms":b.get('p50_tpot_ms'),"tpot_p95_ms":b.get('p95_tpot_ms'),
                 "tpot_p99_ms":b.get('p99_tpot_ms'),"e2e_p50_ms":b.get('p50_e2el_ms'),
                 "e2e_p95_ms":b.get('p95_e2el_ms'),"e2e_p99_ms":b.get('p99_e2el_ms')}
            for eng in ('decode_a','decode_b'):
                values=[x.get('engines',{}).get(eng,{}) for x in samples]
                row[f'{eng}_running_avg']=stat([x.get('num_requests_running') for x in values])['avg']
                row[f'{eng}_running_max']=stat([x.get('num_requests_running') for x in values])['max']
                row[f'{eng}_waiting_avg']=stat([x.get('num_requests_waiting') for x in values])['avg']
                row[f'{eng}_waiting_max']=stat([x.get('num_requests_waiting') for x in values])['max']
                row[f'{eng}_token_delta']=counter_delta(samples,eng,'generation_tokens_total')
                proc=[x.get('processes',{}).get(eng,{}) for x in samples]
                row[f'{eng}_cpu_avg']=stat([x.get('cpu_percent') for x in proc])['avg']
                row[f'{eng}_cpu_p95']=stat([x.get('cpu_percent') for x in proc])['p95']
            for npu in range(12,16):
                values=[x.get('npus',{}).get(str(npu),{}) for x in samples]
                row[f'npu{npu}_aicore_avg']=stat([x.get('aicore_percent') for x in values])['avg']
                row[f'npu{npu}_aicore_p95']=stat([x.get('aicore_percent') for x in values])['p95']
                row[f'npu{npu}_hbm_max_mib']=stat([x.get('hbm_used_mib') for x in values])['max']
            rows.append(row)
        mode_summary={"mode":mode_dir.name,"samples":len(observations),"engines":{},"processes":{},"npus":{}}
        for eng in ('decode_a','decode_b'):
            vals=[x.get('engines',{}).get(eng,{}) for x in observations]
            mode_summary['engines'][eng]={"running":stat([x.get('num_requests_running') for x in vals]),
                "waiting":stat([x.get('num_requests_waiting') for x in vals]),
                "generation_token_delta":counter_delta(observations,eng,'generation_tokens_total')}
        for proc in ('decode_a','decode_b'):
            vals=[x.get('processes',{}).get(proc,{}) for x in observations]
            mode_summary['processes'][proc]={"cpu_percent":stat([x.get('cpu_percent') for x in vals]),
                                             "rss_mib":stat([x.get('rss_mib') for x in vals])}
        for npu in range(12,16):
            vals=[x.get('npus',{}).get(str(npu),{}) for x in observations]
            mode_summary['npus'][str(npu)]={"aicore_percent":stat([x.get('aicore_percent') for x in vals]),
                "hbm_used_mib":stat([x.get('hbm_used_mib') for x in vals])}
        probe=mode_dir/'consistency.json'
        if probe.exists(): mode_summary['consistency_hashes']=[x['sha256'] for x in json.loads(probe.read_text())['results']]
        details.append(mode_summary)
    with (root/'benchmark_summary.csv').open('w',newline='',encoding='utf-8') as f:
        if rows: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    grouped={}
    for mode in ('D0','D1','D2'):
        grouped[mode]={}
        for c in ('c8','c16'):
            subset=[r for r in rows if r['mode']==mode and f'_{c}_' in r['case']]
            grouped[mode][c]={k:stat([r[k] for r in subset]) for k in ('output_tok_s','goodput_req_s','ttft_p50_ms','ttft_p95_ms','ttft_p99_ms','tpot_p50_ms','tpot_p95_ms','tpot_p99_ms','e2e_p50_ms','e2e_p95_ms','e2e_p99_ms')}
    hashes={d['mode']:d.get('consistency_hashes',[]) for d in details}
    reference=hashes.get('D0',[])
    consistency={mode:{"matches_d0": bool(reference) and value==reference,"count":len(value)} for mode,value in hashes.items()}
    result={"rows":rows,"aggregate":grouped,"telemetry":details,"output_consistency":consistency}
    (root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({"cases":len(rows),"analysis":str(root/'analysis.json'),"consistency":consistency},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
