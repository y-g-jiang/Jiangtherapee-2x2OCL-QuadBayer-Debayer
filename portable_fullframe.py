import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from portable_inference import load_model, demosaic

def positions(length, core, stride):
    if length <= core:
        return [0]
    return sorted(set(range(0, length - core + 1, stride)) | {length - core})

def fade(length, overlap):
    window = np.ones(length, np.float32)
    n = min(overlap, length // 2)
    if n:
        window[:n] = np.sin(np.linspace(0.01, np.pi / 2, n)) ** 2
        window[-n:] = window[:n][::-1]
    return window

def tiled_demosaic(net, counts, black, white, origin, output, core=512, halo=64, stride=448, progress=None):
    counts = np.asarray(counts)
    assert counts.ndim == 2 and counts.size and np.isfinite(counts).all()
    assert white > black and core > 0 and (halo >= 0) and (0 < stride <= core)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    (h, w) = counts.shape
    (ch, cw) = (min(core, h), min(core, w))
    overlap = core - stride
    window = fade(ch, overlap)[:, None] * fade(cw, overlap)[None, :]
    result = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(h, w, 3))
    result[:] = 0
    weight = np.zeros((h, w), np.float32)
    (rows, cols) = (positions(h, ch, stride), positions(w, cw, stride))
    for (row, y) in enumerate(rows):
        for x in cols:
            (y0, x0) = (max(0, y - halo), max(0, x - halo))
            (y1, x1) = (min(h, y + ch + halo), min(w, x + cw + halo))
            pred = demosaic(net, counts[y0:y1, x0:x1], black, white, (origin[0] + y0, origin[1] + x0))
            tile = pred[y - y0:y - y0 + ch, x - x0:x - x0 + cw]
            result[y:y + ch, x:x + cw] += tile * window[:, :, None]
            weight[y:y + ch, x:x + cw] += window
        if progress:
            progress(row + 1, len(rows))
    assert weight.min() > 0
    for y in range(0, h, core):
        result[y:y + core] /= weight[y:y + core, :, None]
        assert np.isfinite(result[y:y + core]).all()
    result.flush()
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['weights', 'input', 'output']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--black', type=float, required=True)
    ap.add_argument('--white', type=float, required=True)
    ap.add_argument('--origin-y', type=int, required=True)
    ap.add_argument('--origin-x', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--core', type=int, default=512)
    ap.add_argument('--halo', type=int, default=64)
    ap.add_argument('--stride', type=int, default=448)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = np.load(args.input, mmap_mode='r', allow_pickle=False)
    counts = source['raw'] if isinstance(source, np.lib.npyio.NpzFile) else source
    net = load_model(args.weights, args.device)
    start = time.time()
    result = tiled_demosaic(net, counts, args.black, args.white, (args.origin_y, args.origin_x), args.output, args.core, args.halo, args.stride, lambda done, total: print(f'Rows {done}/{total}', flush=True))
    record = dict(shape=list(result.shape), seconds=time.time() - start, input=args.input, weights=args.weights, black=args.black, white=args.white, cfa_origin=[args.origin_y, args.origin_x], core=args.core, halo=args.halo, stride=args.stride)
    Path(args.output + '.json').write_text(json.dumps(record, indent=2))
if __name__ == '__main__':
    main()
