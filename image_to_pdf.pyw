"""
이미지 → PDF 변환기
- 이미지 파일 추가: 여러 이미지를 1개 PDF로
- 상위 폴더 선택: 하위 폴더별로 각각 PDF 생성 (저장 위치/상위폴더명/ 안에 저장)
- 이미 변환된 PDF가 있으면 건너뜀
- Real-ESRGAN 업스케일 지원 (realesrgan-ncnn-vulkan.exe 필요)
"""

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

DEFAULT_SAVE_DIR = r"G:\tokki"
DEFAULT_OPEN_DIR = r"G:\만화"

_SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = _SCRIPT_DIR / "models"


def _find_exe():
    # upscayl-bin(최신 ncnn 빌드)이 있으면 우선 사용, 없으면 기존 exe
    for name in ("upscayl-bin.exe", "realesrgan-ncnn-vulkan.exe"):
        p = _SCRIPT_DIR / name
        if p.exists():
            return p
    return _SCRIPT_DIR / "realesrgan-ncnn-vulkan.exe"


ESRGAN_EXE = _find_exe()
# upscayl-bin은 옵션 체계가 다름: -s=출력배율, -z=모델배율, GPU 목록은 -v 필요
IS_UPSCAYL = ESRGAN_EXE.name.lower() == "upscayl-bin.exe"

# PyTorch(CUDA) 백엔드: upscale_torch.py가 있으면 우선 사용 (최신 NVIDIA GPU 지원)
TORCH_SCRIPT = _SCRIPT_DIR / "upscale_torch.py"
USE_TORCH = TORCH_SCRIPT.exists()

ESRGAN_MODELS = ["realesrgan-x4plus-anime", "realesrgan-x4plus", "realesr-animevideov3"]

# 모델별 지원 배율 (x4plus 계열은 4배 전용)
MODEL_SCALES = {
    "realesrgan-x4plus-anime": [4],
    "realesrgan-x4plus": [4],
    "realesr-animevideov3": [2, 3, 4],
}


def images_to_pdf(image_paths, output_path):
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    if not imgs:
        raise ValueError("이미지가 없습니다.")
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:])


# ───────────────────── 만화 폴더 정리기 로직 ─────────────────────
ORG_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
ORG_SPLIT_RE = re.compile(r"^(\d+)-(\d+)_")   # 분할 폴더: NNNN-PP_제목
ORG_MAIN_RE = re.compile(r"^(\d+(?:\.\d+)?)_")  # 단일/외전 폴더: NNNN_제목, NNNN.M_제목


def _org_is_image(name):
    return name.lower().endswith(ORG_IMG_EXT)


def organize_plan(root, series, side_start):
    """실제 변경 없이 작업 계획(merges, side, renames)을 산출."""
    entries = [e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))]

    groups = {}   # base -> [(part_int, folder)]
    singles = []
    for name in entries:
        m = ORG_SPLIT_RE.match(name)
        if m:
            base, part = m.group(1), int(m.group(2))
            groups.setdefault(base, []).append((part, name))
        elif ORG_MAIN_RE.match(name):
            singles.append(name)

    merges = []
    sides = []
    bases_with_side = set()
    for base, parts in groups.items():
        parts.sort()
        main_parts = [(p, f) for p, f in parts if p < side_start]
        side_parts = [(p, f) for p, f in parts if p >= side_start]
        if main_parts:
            merges.append((base, [f for _, f in main_parts]))
        for p, f in side_parts:
            sides.append((base + "." + str(p) + "_" + series, f))
            bases_with_side.add(base)

    merged_targets = {}
    for base, _ in merges:
        suffix = ".0" if base in bases_with_side else ""
        merged_targets[base] = base + suffix + "_" + series

    rename = []
    for name in singles:
        base = ORG_MAIN_RE.match(name).group(1)
        suffix = ".0" if base in bases_with_side else ""
        new = base + suffix + "_" + series
        if new != name:
            rename.append((name, new))

    return merges, merged_targets, sides, rename


def organize_run(root, series, side_start, dry, log):
    merges, merged_targets, sides, rename = organize_plan(root, series, side_start)

    if merges:
        log("■ 분할 합치기 (" + str(len(merges)) + "건)")
    for base, sources in merges:
        target = merged_targets[base]
        tpath = os.path.join(root, target)
        log("    " + ", ".join(sources))
        log("        →  " + target)
        if dry:
            continue
        os.makedirs(tpath, exist_ok=True)
        n = 1
        for src in sources:
            spath = os.path.join(root, src)
            files = sorted(f for f in os.listdir(spath) if _org_is_image(f))
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                dest = os.path.join(tpath, "%03d%s" % (n, ext))
                while os.path.exists(dest):
                    n += 1
                    dest = os.path.join(tpath, "%03d%s" % (n, ext))
                try:
                    shutil.move(os.path.join(spath, f), dest)
                except OSError as e:
                    log("        ! 이동 실패 " + f + ": " + str(e))
                n += 1
            try:
                if src != target:
                    os.rmdir(spath)
            except OSError as e:
                log("   ! 빈 폴더 삭제 실패: " + src + " (" + str(e) + ")")

    if sides:
        log("")
        log("■ 외전 이름변경 (" + str(len(sides)) + "건)")
    for new, src in sides:
        log("    " + src)
        log("        →  " + new)
        if dry:
            continue
        sp, np_ = os.path.join(root, src), os.path.join(root, new)
        if sp == np_:
            continue
        if os.path.exists(np_):
            log("        ! 건너뜀: 대상이 이미 존재함")
            continue
        try:
            os.rename(sp, np_)
        except OSError as e:
            log("        ! 실패: " + str(e))

    if rename:
        log("")
        log("■ 시리즈명 통일 (" + str(len(rename)) + "건)")
    for old, new in rename:
        log("    " + old)
        log("        →  " + new)
        if dry:
            continue
        op, npth = os.path.join(root, old), os.path.join(root, new)
        if os.path.exists(npth):
            log("        ! 건너뜀: 대상이 이미 존재함")
            continue
        try:
            os.rename(op, npth)
        except OSError as e:
            log("        ! 실패: " + str(e))

    total = len(merges) + len(sides) + len(rename)
    log("")
    log(("[미리보기] " if dry else "") + "처리 대상 " + str(total) + "건 "
        "(합치기 " + str(len(merges)) + ", 외전 " + str(len(sides)) +
        ", 이름변경 " + str(len(rename)) + ")")


def list_gpus():
    """exe를 짧게 실행해 사용 가능한 GPU 목록을 파싱한다. [(번호, 이름), ...]"""
    if not ESRGAN_EXE.exists():
        return []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with tempfile.TemporaryDirectory() as td:
            tin = Path(td) / "probe.png"
            tout = Path(td) / "probe_out.png"
            Image.new("RGB", (4, 4), (128, 128, 128)).save(tin)
            cmd = [str(ESRGAN_EXE), "-i", str(tin), "-o", str(tout),
                   "-m", str(MODELS_DIR), "-n", "realesr-animevideov3", "-s", "2"]
            if IS_UPSCAYL:
                cmd += ["-z", "2", "-v"]
            proc = subprocess.run(cmd, capture_output=True,
                                  encoding="utf-8", errors="ignore",
                                  creationflags=flags, timeout=30)
            text = (proc.stderr or "") + (proc.stdout or "")
    except Exception:
        return []
    gpus = {}
    for line in text.splitlines():
        m = re.match(r"\[(\d+)\s+([^\]]+)\]", line.strip())
        if m:
            gpus[int(m.group(1))] = m.group(2).strip()
    return sorted(gpus.items())


def default_gpu(gpus):
    """외장 GPU(NVIDIA/AMD)를 우선 선택, 없으면 첫 번째."""
    for num, name in gpus:
        if any(k in name for k in ("NVIDIA", "GeForce", "RTX", "AMD", "Radeon")):
            return num
    return gpus[0][0] if gpus else None


def _esrgan_run(src_dir, dst_dir, scale, model, tile, gpu, progress_cb, total):
    for p in dst_dir.glob("*.png"):
        p.unlink()
    if USE_TORCH:
        cmd = [sys.executable, str(TORCH_SCRIPT), "-i", str(src_dir), "-o", str(dst_dir),
               "-m", str(MODELS_DIR), "-n", model, "-s", str(scale), "-t", str(tile)]
        if gpu is not None:
            cmd += ["-g", str(gpu)]
    else:
        cmd = [str(ESRGAN_EXE), "-i", str(src_dir), "-o", str(dst_dir),
               "-m", str(MODELS_DIR), "-n", model, "-s", str(scale), "-f", "png", "-t", str(tile)]
        if IS_UPSCAYL:
            cmd += ["-z", str(scale)]  # upscayl: 모델 배율 명시
        if gpu is not None:
            cmd += ["-g", str(gpu)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    log_path = dst_dir / "_esrgan.log"
    with open(log_path, "w", encoding="utf-8", errors="ignore") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf,
                                creationflags=flags)
        while proc.poll() is None:
            if progress_cb and total:
                done = len([p for p in dst_dir.iterdir() if p.suffix.lower() == ".png"])
                progress_cb(min(done, total), total)
            time.sleep(0.3)
    if progress_cb and total:
        progress_cb(total, total)
    if proc.returncode != 0:
        err = log_path.read_text(encoding="utf-8", errors="ignore")
        raise RuntimeError(err or "업스케일 실패")


def _count_black(dst_dir):
    """완전히 검은(픽셀 최대값 0) PNG 수를 센다. (검은 수, 전체 수) 반환.
    만화엔 정상적으로 새까만 페이지가 있을 수 있으므로, '전부' 검을 때만 실패로 본다."""
    pngs = [p for p in dst_dir.iterdir() if p.suffix.lower() == ".png"]
    if not pngs:
        return (0, 0)
    black = 0
    for p in pngs:
        try:
            ex = Image.open(p).convert("RGB").getextrema()
        except Exception:
            black += 1
            continue
        if max(hi for _, hi in ex) == 0:
            black += 1
    return (black, len(pngs))


def upscale_folder(src_dir, dst_dir, scale, model, tile=0, progress_cb=None, status_cb=None, gpus=None):
    if not USE_TORCH and not ESRGAN_EXE.exists():
        raise FileNotFoundError(
            "realesrgan-ncnn-vulkan.exe 를 찾을 수 없습니다.\n"
            "exe 파일을 프로그램과 같은 폴더에 넣어주세요:\n"
            "https://github.com/xinntao/Real-ESRGAN/releases"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = len([p for p in src_dir.iterdir() if p.suffix.lower() in SUPPORTED])

    if not gpus:
        gpus = [None]

    # 결과가 전부 검으면(= 해당 GPU가 처리 실패) 다음 GPU로 자동 폴백
    last = (0, 0)
    for gi, g in enumerate(gpus):
        if gi > 0 and status_cb:
            status_cb("결과 전부 검음 → 다른 GPU로 재시도 (GPU " + str(g) + ")")
        _esrgan_run(src_dir, dst_dir, scale, model, tile, g, progress_cb, total)
        nb, nt = _count_black(dst_dir)
        last = (nb, nt)
        if nb < nt:  # 일부만 검으면(=정상 검은 페이지) 통과
            return g
    raise RuntimeError("모든 GPU에서 업스케일 결과가 전부 검게 나옵니다 (" + str(last[0]) + "/" +
                       str(last[1]) + "장). exe가 이 GPU와 호환되지 않을 수 있습니다 — "
                       "realesrgan-ncnn-vulkan을 최신 버전으로 교체해 보세요.")


class TorchUpscaler:
    """upscale_torch.py를 --server 모드로 한 번만 띄워 재사용한다.
    화마다의 파이썬/CUDA/모델 초기화 비용을 제거해 배치 처리를 빠르게 한다."""

    def __init__(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [sys.executable, str(TORCH_SCRIPT), "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="ignore", creationflags=flags)
        ready = self.proc.stdout.readline()
        if "READY" not in (ready or ""):
            raise RuntimeError("PyTorch 업스케일 서버 시작 실패")

    def upscale(self, src, dst, model, scale, tile, progress_cb):
        dst.mkdir(parents=True, exist_ok=True)
        for p in dst.glob("*.png"):
            p.unlink()
        total = len([p for p in src.iterdir() if p.suffix.lower() in SUPPORTED])
        req = json.dumps({"i": str(src), "o": str(dst), "m": str(MODELS_DIR),
                          "n": model, "s": scale, "t": tile})
        self.proc.stdin.write(req + "\n")
        self.proc.stdin.flush()
        result = {}

        def reader():
            result["line"] = self.proc.stdout.readline()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        while th.is_alive():
            if progress_cb and total:
                done = len([p for p in dst.iterdir() if p.suffix.lower() == ".png"])
                progress_cb(min(done, total), total)
            time.sleep(0.3)
        if progress_cb and total:
            progress_cb(total, total)
        line = (result.get("line") or "").strip()
        if line.startswith("ERR"):
            raise RuntimeError(line[3:].strip("\t ") or "업스케일 실패")
        if not line:
            raise RuntimeError("업스케일 서버 응답 없음 (프로세스가 종료됨)")

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("만화 도구 — PDF 변환 / 폴더 정리")
        self.resizable(True, True)
        self.geometry("720x760")
        self._gpus = [] if USE_TORCH else list_gpus()
        self._working_gpu = None
        self._torch = None
        self._q = queue.Queue()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        pdf_tab = ttk.Frame(nb)
        org_tab = ttk.Frame(nb)
        nb.add(pdf_tab, text="이미지 → PDF")
        nb.add(org_tab, text="폴더 정리")
        self._build_ui(pdf_tab)
        self._build_organizer(org_tab)

    def _build_ui(self, parent):
        btn_frame = tk.Frame(parent, pady=6)
        btn_frame.pack(fill="x", padx=10)

        tk.Button(btn_frame, text="이미지 추가", width=14, command=self._add_images).pack(side="left", padx=3)
        tk.Button(btn_frame, text="상위 폴더 선택", width=16, command=self._add_folder).pack(side="left", padx=3)
        tk.Button(btn_frame, text="선택 삭제", width=14, command=self._remove_selected).pack(side="left", padx=3)
        tk.Button(btn_frame, text="위로", width=8, command=self._move_up).pack(side="left", padx=2)
        tk.Button(btn_frame, text="아래로", width=8, command=self._move_down).pack(side="left", padx=2)

        list_frame = tk.Frame(parent)
        list_frame.pack(fill="both", expand=True, padx=10)

        cols = ("type", "name", "path")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("type", text="종류")
        self.tree.heading("name", text="이름")
        self.tree.heading("path", text="경로")
        self.tree.column("type", width=60, stretch=False)
        self.tree.column("name", width=200)
        self.tree.column("path", width=380)

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        out_frame = tk.Frame(parent, pady=6)
        out_frame.pack(fill="x", padx=10)
        tk.Label(out_frame, text="저장 폴더:").pack(side="left")
        self.out_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
        tk.Entry(out_frame, textvariable=self.out_var, width=45).pack(side="left", padx=4)
        tk.Button(out_frame, text="찾기", command=self._pick_out_dir).pack(side="left")

        name_frame = tk.Frame(parent, pady=2)
        name_frame.pack(fill="x", padx=10)
        tk.Label(name_frame, text="PDF 파일명 (이미지 모드):").pack(side="left")
        self.pdf_name_var = tk.StringVar(value="output")
        tk.Entry(name_frame, textvariable=self.pdf_name_var, width=24).pack(side="left", padx=4)
        tk.Label(name_frame, text=".pdf").pack(side="left")

        upscale_frame = tk.LabelFrame(parent, text="업스케일 (Real-ESRGAN)", pady=6)
        upscale_frame.pack(fill="x", padx=10, pady=(4, 0))

        row1 = tk.Frame(upscale_frame)
        row1.pack(fill="x", padx=4)

        self.upscale_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row1, text="업스케일 적용", variable=self.upscale_var,
                       command=self._toggle_upscale).pack(side="left")

        tk.Label(row1, text="배율:").pack(side="left", padx=(14, 2))
        self.scale_var = tk.StringVar(value="2")
        self.scale_cb = ttk.Combobox(row1, textvariable=self.scale_var,
                                     values=["2", "4"], width=4, state="readonly")
        self.scale_cb.pack(side="left")

        tk.Label(row1, text="모델:").pack(side="left", padx=(10, 2))
        self.model_var = tk.StringVar(value="realesr-animevideov3")
        model_values = (["realesr-animevideov3", "realesrgan-x4plus-anime"]
                        if USE_TORCH else ESRGAN_MODELS)
        self.model_cb = ttk.Combobox(row1, textvariable=self.model_var,
                                     values=model_values, width=28, state="readonly")
        self.model_cb.pack(side="left", padx=4)

        row2 = tk.Frame(upscale_frame)
        row2.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(row2, text="타일:").pack(side="left", padx=(0, 2))
        tile_default = "64"
        tile_values = (["1024", "768", "512", "384", "256", "128", "64"] if USE_TORCH
                       else ["0", "400", "200", "100", "64"])
        self.tile_var = tk.StringVar(value=tile_default)
        self.tile_cb = ttk.Combobox(row2, textvariable=self.tile_var,
                                    values=tile_values, width=5, state="readonly")
        self.tile_cb.pack(side="left")
        tile_hint = ("(낮을수록 VRAM↓, 약간 느려짐)" if USE_TORCH
                     else "(깨지면 낮추기, 0=자동)")
        tk.Label(row2, text=tile_hint, fg="gray").pack(side="left", padx=2)

        row3 = tk.Frame(upscale_frame)
        row3.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(row3, text="GPU:").pack(side="left", padx=(0, 2))
        if USE_TORCH:
            gpu_values = []
            default_str = "CUDA 자동 (NVIDIA)"
            gpu_state = "disabled"
            gpu_hint = ""
        elif self._gpus:
            gpu_values = [str(n) + ": " + nm for n, nm in self._gpus]
            default_dev = default_gpu(self._gpus)
            default_str = next((v for v in gpu_values if v.startswith(str(default_dev) + ":")), "")
            gpu_state = "readonly"
            gpu_hint = ""
        else:
            # 자동 감지 실패 → 번호 직접 입력 (0=보통 내장, 1=보통 외장)
            gpu_values = ["0", "1", "2"]
            default_str = "1"
            gpu_state = "normal"
            gpu_hint = "자동 감지 실패 — GPU 번호 직접 선택/입력 (1=보통 외장)"
        self.gpu_var = tk.StringVar(value=default_str)
        self.gpu_cb = ttk.Combobox(row3, textvariable=self.gpu_var,
                                   values=gpu_values, width=42, state=gpu_state)
        self.gpu_cb.pack(side="left")
        if gpu_hint:
            tk.Label(row3, text=gpu_hint, fg="gray").pack(side="left", padx=4)

        if USE_TORCH:
            exe_status, exe_ok = "PyTorch(CUDA) 백엔드 사용", True
        elif ESRGAN_EXE.exists():
            exe_status, exe_ok = "감지됨: " + ESRGAN_EXE.name, True
        else:
            exe_status, exe_ok = "exe 없음 -- 프로그램과 같은 폴더에 넣어주세요", False
        tk.Label(upscale_frame, text=exe_status,
                 fg="green" if exe_ok else "red").pack(anchor="w", padx=4)

        # ── 진행도 ──────────────────────────────────
        prog_frame = tk.Frame(parent, pady=4)
        prog_frame.pack(fill="x", padx=10)

        status_row = tk.Frame(prog_frame)
        status_row.pack(fill="x")
        self.status_var = tk.StringVar(value="")
        tk.Label(status_row, textvariable=self.status_var, anchor="w").pack(side="left")
        self.chapter_var = tk.StringVar(value="")
        tk.Label(status_row, textvariable=self.chapter_var, anchor="e", fg="#2563eb").pack(side="right")

        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x", pady=(2, 0))

        # ── 변환 버튼 ──────────────────────────────
        self.convert_btn = tk.Button(parent, text="PDF 변환 시작", font=("", 12, "bold"),
                                     bg="#2563eb", fg="white", activebackground="#1d4ed8",
                                     pady=6, command=self._convert)
        self.convert_btn.pack(fill="x", padx=10, pady=(2, 10))

        self._items = {}

    # ───────────────── 폴더 정리 탭 ─────────────────
    def _build_organizer(self, parent):
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="대상 폴더").grid(row=0, column=0, sticky="w")
        self.org_path = tk.StringVar()
        ttk.Entry(frm, textvariable=self.org_path, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="찾아보기...", command=self._org_browse).grid(row=0, column=2)

        ttk.Label(frm, text="시리즈명").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.org_series = tk.StringVar()
        ttk.Entry(frm, textvariable=self.org_series, width=60).grid(row=1, column=1, sticky="we", padx=4, pady=(8, 0))
        ttk.Label(frm, text="폴더명 자동", foreground="#666").grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="외전 시작 파트번호").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.org_side = tk.IntVar(value=5)
        ttk.Spinbox(frm, from_=2, to=20, textvariable=self.org_side, width=6).grid(row=2, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Label(frm, text="(이 번호 이상은 외전 .5/.6 으로 처리)", foreground="#666").grid(row=3, column=1, sticky="w", padx=4)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, pady=10, sticky="w")
        ttk.Button(btns, text="미리보기", command=lambda: self._org_go(True)).pack(side="left")
        ttk.Button(btns, text="실행", command=lambda: self._org_go(False)).pack(side="left", padx=6)

        self.org_log = tk.Text(frm, height=16, wrap="none")
        self.org_log.grid(row=5, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(frm, command=self.org_log.yview)
        sb.grid(row=5, column=3, sticky="ns")
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=self.org_log.xview)
        hsb.grid(row=6, column=0, columnspan=3, sticky="we")
        self.org_log["yscrollcommand"] = sb.set
        self.org_log["xscrollcommand"] = hsb.set

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

    def _org_browse(self):
        d = filedialog.askdirectory(title="정리할 폴더 선택", initialdir=DEFAULT_OPEN_DIR)
        if d:
            self.org_path.set(d)
            self.org_series.set(os.path.basename(d.rstrip("/\\")))  # 폴더명을 시리즈명으로

    def _org_write(self, s):
        self.org_log.insert("end", s + "\n")
        self.org_log.see("end")
        self.update_idletasks()

    def _org_go(self, dry):
        root = self.org_path.get().strip()
        if not root or not os.path.isdir(root):
            messagebox.showerror("오류", "유효한 폴더를 선택하세요.")
            return
        series = self.org_series.get().strip() or os.path.basename(root.rstrip("/\\"))
        if not series:
            messagebox.showerror("오류", "시리즈명을 알 수 없습니다 (폴더명 확인).")
            return
        if not dry and not messagebox.askyesno(
                "확인", "실제로 폴더를 변경합니다. 진행할까요?\n(먼저 미리보기로 확인하는 것을 권장)"):
            return
        self.org_log.delete("1.0", "end")
        try:
            organize_run(root, series, self.org_side.get(), dry, self._org_write)
        except Exception as e:
            messagebox.showerror("오류", str(e))

    def _toggle_upscale(self):
        state = "readonly" if self.upscale_var.get() else "disabled"
        self.scale_cb.config(state=state)
        self.model_cb.config(state=state)
        self.tile_cb.config(state=state)
        self.gpu_cb.config(state="disabled" if USE_TORCH else state)

    def _selected_gpu(self):
        s = self.gpu_var.get()
        try:
            return int(s.split(":")[0])
        except (ValueError, IndexError):
            return None

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="이미지 선택",
            initialdir=DEFAULT_OPEN_DIR,
            filetypes=[("이미지 파일", " ".join("*" + e for e in SUPPORTED)), ("모든 파일", "*.*")]
        )
        for p in paths:
            self._insert_item("image", Path(p))

    def _add_folder(self):
        d = filedialog.askdirectory(title="상위 폴더 선택", initialdir=DEFAULT_OPEN_DIR)
        if not d:
            return
        parent = Path(d)
        subfolders = sorted([p for p in parent.iterdir() if p.is_dir()])
        if not subfolders:
            messagebox.showwarning("경고", parent.name + " 안에 하위 폴더가 없습니다.")
            return
        # 상위 폴더를 새로 고르면 기존 목록 초기화 후 다시 채움
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._items.clear()
        for sub in subfolders:
            self._insert_item("folder", sub, parent_name=parent.name)

    def _insert_item(self, kind, path, parent_name=""):
        for meta in self._items.values():
            if meta["path"] == path:
                return
        label = "이미지" if kind == "image" else "폴더"
        iid = self.tree.insert("", "end", values=(label, path.name, str(path)))
        self._items[iid] = {"kind": kind, "path": path, "parent_name": parent_name}

    def _remove_selected(self):
        for iid in self.tree.selection():
            self.tree.delete(iid)
            self._items.pop(iid, None)

    def _move_up(self):
        for iid in self.tree.selection():
            idx = self.tree.index(iid)
            if idx > 0:
                self.tree.move(iid, "", idx - 1)

    def _move_down(self):
        for iid in reversed(self.tree.selection()):
            idx = self.tree.index(iid)
            if idx < len(self.tree.get_children()) - 1:
                self.tree.move(iid, "", idx + 1)

    def _pick_out_dir(self):
        d = filedialog.askdirectory(title="저장 폴더 선택")
        if d:
            self.out_var.set(d)

    def _get_upscaled_images(self, src_dir, tmp_root):
        tmp_out = tmp_root / src_dir.name
        if USE_TORCH:
            if self._torch is None:
                self._torch = TorchUpscaler()
            self._torch.upscale(src_dir, tmp_out, self.model_var.get(),
                                int(self.scale_var.get()), int(self.tile_var.get()),
                                lambda d, t: self._q.put(("chapter", (d, t))))
        else:
            # 시도 순서: 직전에 성공한 GPU(있으면) → 사용자가 고른 GPU → 나머지 GPU들
            sel = self._working_gpu if self._working_gpu is not None else self._selected_gpu()
            order = []
            if sel is not None:
                order.append(sel)
            for n, _ in self._gpus:
                if n != sel:
                    order.append(n)
            if not order:
                order = [None]
            used = upscale_folder(src_dir, tmp_out, int(self.scale_var.get()), self.model_var.get(),
                                  int(self.tile_var.get()),
                                  lambda d, t: self._q.put(("chapter", (d, t))),
                                  lambda m: self._q.put(("status", m)),
                                  gpus=order)
            if used is not None:
                self._working_gpu = used
        return sorted([p for p in tmp_out.iterdir() if p.suffix.lower() in SUPPORTED])

    def _convert(self):
        out_dir = Path(self.out_var.get())
        if not out_dir.exists():
            messagebox.showerror("오류", "저장 폴더가 존재하지 않습니다:\n" + str(out_dir))
            return

        ordered_iids = self.tree.get_children()
        if not ordered_iids:
            messagebox.showwarning("경고", "변환할 항목이 없습니다.")
            return

        if self.upscale_var.get() and not USE_TORCH:
            model = self.model_var.get()
            scale = int(self.scale_var.get())
            allowed = MODEL_SCALES.get(model)
            if allowed and scale not in allowed:
                messagebox.showerror(
                    "모델·배율 불일치",
                    "'" + model + "' 모델은 배율 " + str(allowed) + " 만 지원합니다.\n"
                    "현재 배율: " + str(scale) + "\n\n"
                    "배율을 바꾸거나 다른 모델을 선택하세요.\n"
                    "(2배를 쓰려면 realesr-animevideov3 모델을 선택하세요.)"
                )
                return

        image_paths = []
        folder_items = []
        for iid in ordered_iids:
            meta = self._items[iid]
            if meta["kind"] == "image":
                image_paths.append(meta["path"])
            else:
                folder_items.append(meta)

        total = (1 if image_paths else 0) + len(folder_items)
        self.progress["maximum"] = total
        self.progress["value"] = 0
        self.convert_btn.config(state="disabled")

        def worker():
            results = []
            errors = []
            done = 0

            with tempfile.TemporaryDirectory() as tmp_root:
                tmp_root_path = Path(tmp_root)

                if image_paths:
                    pdf_name = self.pdf_name_var.get().strip() or "output"
                    if not pdf_name.endswith(".pdf"):
                        pdf_name += ".pdf"
                    out_path = out_dir / pdf_name
                    self._q.put(("status", "변환 중: " + pdf_name))
                    try:
                        if self.upscale_var.get():
                            tmp_src = tmp_root_path / "_images_src"
                            tmp_src.mkdir()
                            for p in image_paths:
                                shutil.copy(p, tmp_src / p.name)
                            src = self._get_upscaled_images(tmp_src, tmp_root_path)
                        else:
                            src = image_paths
                        images_to_pdf(src, out_path)
                        results.append("OK " + out_path.name + " (" + str(len(src)) + "장)")
                    except Exception as e:
                        errors.append("실패 이미지 PDF: " + str(e))
                    done += 1
                    self._q.put(("progress", done))

                for item in folder_items:
                    folder = item["path"]
                    parent_name = item["parent_name"]
                    save_dir = out_dir / parent_name if parent_name else out_dir
                    save_dir.mkdir(exist_ok=True)
                    out_path = save_dir / (folder.name + ".pdf")

                    if out_path.exists():
                        results.append("건너뜀 " + out_path.name)
                        done += 1
                        self._q.put(("progress", done))
                        continue

                    self._q.put(("status", "변환 중: " + folder.name))
                    try:
                        imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED])
                        if not imgs:
                            errors.append("이미지 없음: " + folder.name)
                            done += 1
                            self._q.put(("progress", done))
                            continue
                        if self.upscale_var.get():
                            imgs = self._get_upscaled_images(folder, tmp_root_path)
                        images_to_pdf(imgs, out_path)
                        tag = " [" + self.scale_var.get() + "x]" if self.upscale_var.get() else ""
                        results.append("OK " + out_path.name + " (" + str(len(imgs)) + "장)" + tag)
                    except Exception as e:
                        errors.append("실패 " + folder.name + ": " + str(e))
                    done += 1
                    self._q.put(("progress", done))

            if self._torch is not None:
                self._torch.close()
                self._torch = None

            if errors:
                try:
                    log_file = out_dir / "변환_실패_로그.txt"
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write("=== " + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
                        for e in errors:
                            f.write(e + "\n")
                        f.write("\n")
                except Exception:
                    pass

            self._q.put(("done", (results, errors)))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_queue(total)

    def _show_result(self, title, lines):
        """항목이 많을 때 화면 밖으로 안 나가도록 스크롤 가능한 결과창."""
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("560x440")
        ok = len([x for x in lines if x.startswith("OK") or x.startswith("건너뜀")])
        bad = len(lines) - ok
        tk.Label(win, text=title + "  —  성공/건너뜀 " + str(ok) + ", 실패 " + str(bad),
                 anchor="w", pady=4).pack(fill="x", padx=8)
        tk.Button(win, text="확인", command=win.destroy).pack(side="bottom", pady=6)
        txt = tk.Text(win, wrap="none")
        sb = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 4))
        txt.insert("end", "\n".join(lines))
        txt.config(state="disabled")
        win.transient(self)
        win.grab_set()

    def _poll_queue(self, total):
        try:
            while True:
                msg, data = self._q.get_nowait()
                if msg == "status":
                    self.status_var.set(data)
                    self.chapter_var.set("현재 화: 0%")
                elif msg == "chapter":
                    done, ctotal = data
                    pct = int(done / ctotal * 100) if ctotal else 0
                    self.chapter_var.set("현재 화: " + str(pct) + "% (" +
                                         str(done) + "/" + str(ctotal) + "장)")
                elif msg == "progress":
                    self.progress["value"] = data
                    self.status_var.set(str(data) + " / " + str(total) + " 완료")
                elif msg == "done":
                    results, errors = data
                    self.progress["value"] = total
                    self.status_var.set("완료!")
                    self.convert_btn.config(state="normal")
                    lines = results + errors
                    if errors and not results:
                        title, icon = "변환 실패", messagebox.showerror
                    elif errors:
                        title, icon = "일부 실패", messagebox.showwarning
                    else:
                        title, icon = "완료!", messagebox.showinfo
                    if len(lines) <= 12:
                        icon(title, "\n".join(lines))
                    else:
                        self._show_result(title, lines)
                    return
        except queue.Empty:
            pass
        self.after(100, lambda: self._poll_queue(total))


if __name__ == "__main__":
    App().mainloop()
