"""
PyTorch(CUDA) 업스케일러 — ncnn realesrgan-ncnn-vulkan과 동일한 CLI로 동작.
RTX 50시리즈(Blackwell) 등 최신 NVIDIA GPU 지원.

사용:
  python upscale_torch.py -i <입력폴더|파일> -o <출력폴더|파일> -m <models폴더>
                          -n <모델명> -s <출력배율> [-t 타일] [-g gpu]

모델명 → 가중치(.pth) 매핑은 아래 WEIGHTS 참고.
출력은 항상 PNG로 저장(입력 파일명 stem + .png).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spandrel import ModelLoader

torch.backends.cudnn.benchmark = True

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

WEIGHTS = {
    "realesr-animevideov3": "realesr-animevideov3.pth",
    "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B.pth",
    "realesrgan-x4plus": "RealESRGAN_x4plus.pth",
}


def load_model(name, models_dir, device):
    fname = WEIGHTS.get(name, name + ".pth")
    path = Path(models_dir) / fname
    if not path.exists():
        raise FileNotFoundError("가중치를 찾을 수 없습니다: " + str(path))
    desc = ModelLoader().load_from_file(str(path))
    desc.to(device).eval()
    return desc, int(desc.scale)


@torch.inference_mode()
def upscale_tensor(model, img, native_scale, tile, overlap, device):
    """img: (1,3,H,W) float[0,1] on device → (1,3,H*native,W*native)."""
    _, _, h, w = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            return model(img).float().clamp(0, 1)

    out = torch.zeros((1, 3, h * native_scale, w * native_scale), device=device)
    weight = torch.zeros_like(out)
    step = max(tile - overlap, 1)
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y1, x1 = max(y2 - tile, 0), max(x2 - tile, 0)
            patch = img[:, :, y1:y2, x1:x2]
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                sr = model(patch).float().clamp(0, 1)
            oy1, oy2 = y1 * native_scale, y2 * native_scale
            ox1, ox2 = x1 * native_scale, x2 * native_scale
            out[:, :, oy1:oy2, ox1:ox2] += sr
            weight[:, :, oy1:oy2, ox1:ox2] += 1
            if y2 >= h:
                break
        if x2 >= w and y2 >= h:
            pass
    return out / weight.clamp(min=1)


def process_image(model, native_scale, src, dst, out_scale, tile, overlap, device):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
    sr = upscale_tensor(model, t, native_scale, tile, overlap, device)
    # 원하는 출력 배율로 리사이즈 (예: 4배 모델 → 2배 출력)
    target_h, target_w = h * out_scale, w * out_scale
    if sr.shape[2] != target_h or sr.shape[3] != target_w:
        mode = "area" if target_h < sr.shape[2] else "bicubic"
        sr = F.interpolate(sr, size=(target_h, target_w), mode=mode,
                           align_corners=False if mode == "bicubic" else None).clamp(0, 1)
    arr = (sr.squeeze(0).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    Image.fromarray(arr).save(dst)
    if device == "cuda":
        del t, sr
        torch.cuda.empty_cache()


def _process_folder(model, native_scale, inp, outp, out_scale, tile, device):
    outp.mkdir(parents=True, exist_ok=True)
    tile = tile if tile and tile > 0 else 512
    overlap = min(32, tile // 4)  # 작은 타일이면 오버랩도 줄여 효율 유지
    files = sorted(p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED)
    for p in files:
        process_image(model, native_scale, p, outp / (p.stem + ".png"),
                      out_scale, tile, overlap, device)


def run_server():
    """stdin으로 JSON 요청을 받아 처리하는 상주 서버 모드.
    모델을 한 번만 로딩해서 화마다의 초기화 비용을 제거한다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            sys.stdout.write("ERR\t잘못된 요청\n")
            sys.stdout.flush()
            continue
        if req.get("cmd") == "quit":
            break
        try:
            name = req["n"]
            if name not in cache:
                cache[name] = load_model(name, req["m"], device)
            model, native_scale = cache[name]
            _process_folder(model, native_scale, Path(req["i"]), Path(req["o"]),
                            int(req["s"]), int(req.get("t", 0)), device)
            sys.stdout.write("OK\n")
        except Exception as e:
            sys.stdout.write("ERR\t" + str(e).replace("\n", " ") + "\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("-m", default="models")
    ap.add_argument("-n", default="realesr-animevideov3")
    ap.add_argument("-s", type=int, default=2)        # 출력 배율
    ap.add_argument("-t", type=int, default=0)        # 타일(<=0 또는 작으면 자동)
    ap.add_argument("-g", type=int, default=0)        # ncnn 호환용(토치는 cuda 단일 장치 사용)
    args, _ = ap.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("경고: CUDA를 사용할 수 없어 CPU로 실행합니다(매우 느림).", file=sys.stderr)

    model, native_scale = load_model(args.n, args.m, device)

    # 타일: 0/미지정이면 512. 작은 타일이면 오버랩도 줄여 효율 유지.
    tile = args.t if args.t and args.t > 0 else 512
    overlap = min(32, tile // 4)

    inp, outp = Path(args.i), Path(args.o)
    if inp.is_dir():
        outp.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in SUPPORTED)
        for p in files:
            process_image(model, native_scale, p, outp / (p.stem + ".png"),
                          args.s, tile, overlap, device)
    else:
        outp.parent.mkdir(parents=True, exist_ok=True)
        process_image(model, native_scale, inp, outp, args.s, tile, overlap, device)


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_server()
    else:
        main()
