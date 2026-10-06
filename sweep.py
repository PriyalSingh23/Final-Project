#!/usr/bin/env python3
"""Train/test sweep: resumes training in chunks, runs the full test battery
after each chunk, and logs all four metrics to a CSV. When a snapshot passes
every metric, it is saved as <ckpt>.best.pt and the sweep stops.

Usage (your PC, runs by itself) -- pass the GOLD data dir via --data, e.g.
    python sweep.py --ckpt lpi_checkpoint_v2_607.pt --max-epoch 680 --n-samples 50000

Single chunk (used when driven externally):
    python sweep.py --once --ckpt ckpt_test.pt --chunk 2
"""
import argparse, csv, os, re, subprocess, sys, time

PY = sys.executable

def current_epoch(ckpt):
    import torch
    return torch.load(ckpt, map_location='cpu', weights_only=False)['epoch']

def run_chunk(ckpt, target_epoch, n_samples, data):
    cmd = [PY, 'train.py', '--out', ckpt, '--resume',
           '--epochs', str(target_epoch), '--n-samples', str(n_samples), '--radioml']
    if data:
        cmd += ['--data', data]
    print(f"[sweep] training to epoch {target_epoch} ...", flush=True)
    subprocess.run(cmd, check=True)

def run_tests(ckpt):
    print("[sweep] running test battery ...", flush=True)
    out = subprocess.run([PY, 'test_metrics.py', '--ckpt', ckpt],
                         capture_output=True, text=True, check=True).stdout
    row = {}
    m = re.search(r'KS p-value:\s*([\d.]+)\s+(\w+)', out);          row['ks_p'], row['ks'] = float(m.group(1)), m.group(2)
    m = re.search(r'ratio\s*([\d.]+)x', out);                       row['cyclo_ratio'] = float(m.group(1))
    m = re.search(r'Adversary accuracy:\s*([\d.]+)%\s+(\w+)', out); row['adv_acc'], row['adv'] = float(m.group(1)), m.group(2)
    m = re.search(r'BER:\s*([\d.]+)%\s+(\w+)', out);                row['ber_pct'], row['ber'] = float(m.group(1)), m.group(2)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='lpi_checkpoint.pt')
    ap.add_argument('--chunk', type=int, default=2)
    ap.add_argument('--max-epoch', type=int, default=680)
    ap.add_argument('--n-samples', type=int, default=50000)
    ap.add_argument('--data', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--once', action='store_true', help='single chunk then exit')
    args = ap.parse_args()
    log = args.log or (args.ckpt + '.sweep.csv')

    new = not os.path.exists(log)
    with open(log, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'ks_p', 'ks', 'cyclo_ratio',
                                          'adv_acc', 'adv', 'ber_pct', 'ber', 'all_pass', 'elapsed_s'])
        if new:
            w.writeheader()
        while True:
            ep = current_epoch(args.ckpt)
            if ep >= args.max_epoch:
                print(f"[sweep] reached max epoch {args.max_epoch}, stopping"); return
            import shutil
            shutil.copy(args.ckpt, args.ckpt + '.prev.pt')   # collapse guard
            t0 = time.time()
            run_chunk(args.ckpt, min(ep + args.chunk, args.max_epoch), args.n_samples, args.data)
            ep = current_epoch(args.ckpt)
            row = run_tests(args.ckpt)
            row['epoch'] = ep
            row['all_pass'] = 'PASS' if (row['ks'] == 'PASS' and row['adv'] == 'PASS'
                                         and row['ber'] == 'PASS') else 'FAIL'
            row['elapsed_s'] = int(time.time() - t0)
            w.writerow(row); f.flush()
            print(f"[sweep] ep {ep}: KS p={row['ks_p']:.4f} {row['ks']} | "
                  f"cyclo {row['cyclo_ratio']}x | adv {row['adv_acc']}% {row['adv']} | "
                  f"BER {row['ber_pct']}% {row['ber']} => {row['all_pass']}", flush=True)
            if row['all_pass'] == 'PASS':
                import shutil
                best = args.ckpt + '.best.pt'
                shutil.copy(args.ckpt, best)
                print(f"[sweep] *** ALL PASS at epoch {ep} -- saved {best} ***")
                return
            if args.once:
                return

if __name__ == '__main__':
    main()
