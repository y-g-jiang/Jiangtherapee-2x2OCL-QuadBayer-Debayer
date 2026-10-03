import sys, hashlib, argparse
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/samsung'))
from arch.simple_multi_head_jd_model import SIMP_MULTI_JD

def load_model(path, device='cpu', expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, 'Weight checksum mismatch'
    cp = torch.load(path, map_location='cpu', weights_only=True)
    state = cp['model']
    net = SIMP_MULTI_JD()
    net.load_state_dict(state, strict=True)
    assert sum((p.numel() for p in net.parameters())) == 3203107, 'Unexpected architecture'
    return net.to(device).eval()

def quad_mask(h, w, origin):
    y = (np.arange(h) + int(origin[0])) // 2 % 2
    x = (np.arange(w) + int(origin[1])) // 2 % 2
    c = np.where(y[:, None] == 0, np.where(x[None, :] == 0, 0, 1), np.where(x[None, :] == 0, 1, 2))
    return np.stack([c == i for i in range(3)]).astype(np.float32)

@torch.inference_mode()
def predict_tensor(net, raw, mask):
    assert raw.ndim == 4 and raw.shape[1] == 1 and (mask.shape == (raw.shape[0], 3, *raw.shape[2:]))
    assert raw.dtype == mask.dtype == torch.float32
    x = torch.cat((raw, mask), 1) * 2 - 1
    x = net.relu(net.first_conv(x))
    x = net.relu(net.first_conv_upsample(x))
    return (net.backbone(x)[0] + 1) * 0.5

def demosaic(net, raw_counts, black, white, cfa_origin):
    a = np.asarray(raw_counts)
    assert a.ndim == 2 and white > black and np.isfinite(a).all()
    raw = (a.astype(np.float32) - black) / (white - black)
    mask = quad_mask(*a.shape, cfa_origin)
    device = next(net.parameters()).device
    result = predict_tensor(net, torch.from_numpy(raw[None, None]).to(device), torch.from_numpy(mask[None]).to(device))[0].cpu().numpy().transpose(1, 2, 0)
    assert np.isfinite(result).all()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--black', type=float, required=True)
    ap.add_argument('--white', type=float, required=True)
    ap.add_argument('--origin-y', type=int, required=True)
    ap.add_argument('--origin-x', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    z = np.load(a.input)
    raw = z['raw'] if isinstance(z, np.lib.npyio.NpzFile) else z
    net = load_model(a.weights, a.device)
    result = demosaic(net, raw, a.black, a.white, (a.origin_y, a.origin_x))
    np.save(a.output, result)
if __name__ == '__main__':
    main()
