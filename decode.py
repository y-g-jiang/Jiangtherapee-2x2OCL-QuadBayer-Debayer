import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from portable_fullframe import tiled_demosaic
from portable_inference import load_model, quad_mask

ROOT = Path(__file__).resolve().parent


def tag_values(page, code):
    tag = page.tags[code]
    values = np.asarray(tag.value, dtype=np.float64).reshape(-1)
    if int(tag.dtype) in (5, 10):
        values = values.reshape(-1, 2)
        values = values[:, 0] / values[:, 1]
    if not np.isfinite(values).all():
        raise ValueError(code)
    return values


def scalar_tag(page, code):
    values = tag_values(page, code)
    if not np.all(values == values[0]):
        raise ValueError(code)
    return float(values[0])


def pages(collection):
    for page in collection:
        yield page
        if page.pages is not None:
            yield from pages(page.pages)


def read_input(path, preset):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        return raw, preset['black'], preset['white'], tuple(preset['cfa_origin'])
    if suffix == '.npz':
        with np.load(path, allow_pickle=False) as source:
            raw = source['raw']
        return raw, preset['black'], preset['white'], tuple(preset['cfa_origin'])
    if suffix != '.dng':
        raise ValueError(suffix)
    with tifffile.TiffFile(path) as source:
        candidates = [p for p in pages(source.pages)
                      if len(p.shape) == 2 and 33421 in p.tags and 33422 in p.tags]
        if len(candidates) != 1:
            raise ValueError('CFA IFD')
        page = candidates[0]
        if tuple(page.tags[33421].value) != (4, 4):
            raise ValueError('CFARepeatPatternDim')
        pattern = np.array(list(page.tags[33422].value)).reshape(4, 4)
        origins = [(y, x) for y in range(4) for x in range(4)
                   if np.array_equal(quad_mask(4, 4, (y, x)).argmax(0), pattern)]
        if len(origins) != 1:
            raise ValueError('CFAPattern')
        raw = page.asarray()
        black = scalar_tag(page, 50714)
        white = scalar_tag(page, 50717)
    return raw, black, white, origins[0]


def write_rgb16(linear_path, output):
    source = np.load(linear_path, mmap_mode='r', allow_pickle=False)
    result = np.empty(source.shape, dtype=np.uint16)
    for y in range(0, source.shape[0], 128):
        result[y:y + 128] = np.rint(np.clip(source[y:y + 128], 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(output, result, photometric='rgb', metadata=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preset', default=str(ROOT / 'presets/ov50x_native50.json'))
    parser.add_argument('--weights')
    parser.add_argument('--sha256')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--black', type=float)
    parser.add_argument('--white', type=float)
    parser.add_argument('--origin-y', type=int)
    parser.add_argument('--origin-x', type=int)
    parser.add_argument('--core', type=int)
    parser.add_argument('--halo', type=int)
    parser.add_argument('--stride', type=int)
    args = parser.parse_args()
    preset = json.loads(Path(args.preset).read_text())
    destination = Path(args.output)
    if destination.suffix.lower() not in ('.npy', '.tif', '.tiff'):
        parser.error('--output: .npy/.tif/.tiff')
    if destination.exists():
        raise FileExistsError(destination)
    if (args.origin_y is None) != (args.origin_x is None):
        parser.error('--origin-y + --origin-x')
    raw, black, white, origin = read_input(args.input, preset)
    black = black if args.black is None else args.black
    white = white if args.white is None else args.white
    origin = origin if args.origin_y is None else (args.origin_y, args.origin_x)
    weights = Path(args.weights) if args.weights else ROOT / preset['weights']
    expected = args.sha256 if args.weights else preset['sha256']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = load_model(weights, args.device, expected)
    linear_path = destination if destination.suffix.lower() == '.npy' else destination.with_suffix('.linear.npy')
    tiled_demosaic(net, raw, black, white, origin, linear_path,
                  preset['core'] if args.core is None else args.core,
                  preset['halo'] if args.halo is None else args.halo,
                  preset['stride'] if args.stride is None else args.stride)
    if destination.suffix.lower() != '.npy':
        write_rgb16(linear_path, destination)


if __name__ == '__main__':
    main()
