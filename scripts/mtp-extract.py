#!/usr/bin/env python3
"""Extract an embedded MTP (nextn) draft layer from a qwen4exp GGUF into a
standalone head GGUF that `q4 --mtp` can load.

Models like Qwen3.8-Flash-Next ship the draft block inside the main file as
`blk.<N>.*` tensors (one extra layer past the regular stack). This copies the
GGUF metadata verbatim and repacks only the draft-layer tensors, producing a
~3 GiB head file instead of pointing the loader at the whole 80+ GiB model.

Usage:
  scripts/mtp-extract.py <model.gguf> [-o head.gguf] [--layer N]

Multi-shard inputs are supported: pass shard 00001 and siblings are found.
"""
import argparse
import os
import re
import struct
import sys

ALIGN = 32
# scalar value-type sizes; 8=STRING (u64 len), 9=ARRAY
VT_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _u32(buf, p):
    return struct.unpack_from('<I', buf, p)[0], p + 4


def _u64(buf, p):
    return struct.unpack_from('<Q', buf, p)[0], p + 8


def _str(buf, p):
    n, p = _u64(buf, p)
    return buf[p:p + n].decode('utf-8', 'replace'), p + n


def skip_value(buf, p, vt):
    """Return position just past a kv value of type vt."""
    if vt == 8:
        n, p = _u64(buf, p)
        return p + n
    if vt == 9:
        et, p = _u32(buf, p)
        n, p = _u64(buf, p)
        if et == 8:
            for _ in range(n):
                ln, p = _u64(buf, p)
                p += ln
            return p
        return p + n * VT_SIZE[et]
    return p + VT_SIZE[vt]


def read_value(buf, p, vt):
    """Parse a kv value; returns (value, new_pos). Only scalar int/str used."""
    if vt == 8:
        return _str(buf, p)
    if vt in VT_SIZE:
        fmt = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f',
               7: '<?', 10: '<Q', 11: '<q', 12: '<d'}[vt]
        v = struct.unpack_from(fmt, buf, p)[0]
        return v, p + VT_SIZE[vt]
    # ARRAY: skip, keep raw
    p2 = skip_value(buf, p, vt)
    return None, p2


def parse_gguf(path):
    """Return dict with header bytes, kv region, tensor infos, data start."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(min(size, 1 << 24))
    if head[:4] != b'GGUF':
        raise SystemExit(f'{path}: not a GGUF file')
    ver = _u32(head, 4)[0]
    if ver < 2 or ver > 3:
        raise SystemExit(f'{path}: GGUF version {ver} unsupported')
    n_tensors, n_kv = struct.unpack_from('<QQ', head, 8)
    p = 24
    kv_start = p
    alignment = ALIGN
    for _ in range(n_kv):
        k, p = _str(head, p)
        vt, p = _u32(head, p)
        if k == 'general.alignment':
            v, p = read_value(head, p, vt)
            if v:
                alignment = v
        else:
            p = skip_value(head, p, vt)
    kv_end = p
    tensors = []
    for _ in range(n_tensors):
        name, p = _str(head, p)
        nd, p = _u32(head, p)
        dims = struct.unpack_from('<' + 'Q' * nd, head, p)
        p += 8 * nd
        ty, p = _u32(head, p)
        off, p = _u64(head, p)
        tensors.append({'name': name, 'dims': dims, 'type': ty, 'off': off})
    data_start = ((p + alignment - 1) // alignment) * alignment
    if data_start > len(head):
        # header larger than the read window (huge kv); re-read
        with open(path, 'rb') as f:
            f.seek(0)
            head = f.read(data_start)
    return {'path': path, 'ver': ver, 'n_kv': n_kv,
            'kv': head[kv_start:kv_end], 'infos_end': p,
            'data_start': data_start, 'align': alignment,
            'tensors': tensors, 'size': size}


def tensor_bytes(parts):
    """Compute each tensor's byte size via next-offset / EOF within its file."""
    sizes = {}
    for part in parts:
        ts = sorted(part['tensors'], key=lambda t: t['off'])
        ds, fs = part['data_start'], part['size']
        for i, t in enumerate(ts):
            end = ts[i + 1]['off'] if i + 1 < len(ts) else fs - ds
            sizes[(part['path'], t['name'])] = end - t['off']
    return sizes


def discover_shards(path):
    m = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path)
    if not m:
        return [path]
    tot = int(m.group(2))
    base = path[:m.start()]
    out = [f'{base}-{i:05d}-of-{tot:05d}.gguf' for i in range(1, tot + 1)]
    for p in out:
        if not os.path.exists(p):
            raise SystemExit(f'missing shard {p}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--layer', type=int, default=None,
                    help='draft layer index (default: layer owning .nextn.)')
    a = ap.parse_args()

    parts = [parse_gguf(p) for p in discover_shards(a.model)]
    sizes = tensor_bytes(parts)

    layer = a.layer
    if layer is None:
        pat = re.compile(r'^blk\.(\d+)\.nextn\.')
        cand = [int(pat.match(t['name']).group(1))
                for part in parts for t in part['tensors']
                if pat.match(t['name'])]
        if not cand:
            raise SystemExit('no .nextn. tensors found; pass --layer')
        layer = max(cand)
    prefix = f'blk.{layer}.'
    sel = []
    for part in parts:
        for t in part['tensors']:
            if t['name'].startswith(prefix):
                sel.append((part, t))
    if not sel:
        raise SystemExit(f'no tensors match {prefix}*')
    sel.sort(key=lambda st: st[1]['off'])

    align = parts[0]['align']
    head0 = parts[0]
    out = open(a.out, 'wb')
    # header: magic ver n_tensors n_kv
    out.write(b'GGUF')
    out.write(struct.pack('<IQQ', head0['ver'], len(sel), head0['n_kv']))
    kv_off = out.tell()
    out.write(head0['kv'])
    infos = b''
    off = 0
    new = []
    for part, t in sel:
        n = sizes[(part['path'], t['name'])]
        infos += struct.pack('<Q', len(t['name'])) + t['name'].encode()
        infos += struct.pack('<I', len(t['dims']))
        infos += struct.pack('<' + 'Q' * len(t['dims']), *t['dims'])
        infos += struct.pack('<IQ', t['type'], off)
        new.append((part, t, off, n))
        off += ((n + align - 1) // align) * align
    out.write(infos)
    pad = (-out.tell()) % align
    out.write(b'\0' * pad)
    written = 0
    for part, t, o, n in new:
        want = out.tell()
        tgt = o
        if want < tgt:
            out.write(b'\0' * (tgt - want))
        with open(part['path'], 'rb') as f:
            f.seek(part['data_start'] + t['off'])
            left = n
            while left:
                chunk = f.read(min(left, 1 << 24))
                if not chunk:
                    raise SystemExit(f'short read on {t["name"]}')
                out.write(chunk)
                left -= len(chunk)
        written += n
        print(f"  {t['name']:<64} {n/1e6:9.1f} MB")
    out.close()
    print(f'{a.out}: {len(sel)} tensors from layer {layer}, '
          f'{os.path.getsize(a.out)/2**30:.2f} GiB')


if __name__ == '__main__':
    main()
