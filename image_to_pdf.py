"""
이미지 → PDF 변환기
- 이미지 파일 추가: 여러 이미지를 1개 PDF로
- 상위 폴더 선택: 하위 폴더별로 각각 PDF 생성 (저장 위치/상위폴더명/ 안에 저장)
- 이미 변환된 PDF가 있으면 건너뜀
- Real-ESRGAN 업스케일 지원 (realesrgan-ncnn-vulkan.exe 필요)
"""

import queue
import re
import shutil
import subprocess
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
    if not ESRGAN_EXE.exists():
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 → PDF 변환기")
        self.resizable(True, True)
        self.geometry("700x710")
        self._gpus = list_gpus()
        self._working_gpu = None
        self._build_ui()
        self._q = queue.Queue()

    def _build_ui(self):
        btn_frame = tk.Frame(self, pady=6)
        btn_frame.pack(fill="x", padx=10)

        tk.Button(btn_frame, text="이미지 추가", width=14, command=self._add_images).pack(side="left", padx=3)
        tk.Button(btn_frame, text="상위 폴더 선택", width=16, command=self._add_folder).pack(side="left", padx=3)
        tk.Button(btn_frame, text="선택 삭제", width=14, command=self._remove_selected).pack(side="left", padx=3)
        tk.Button(btn_frame, text="위로", width=8, command=self._move_up).pack(side="left", padx=2)
        tk.Button(btn_frame, text="아래로", width=8, command=self._move_down).pack(side="left", padx=2)

        list_frame = tk.Frame(self)
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

        out_frame = tk.Frame(self, pady=6)
        out_frame.pack(fill="x", padx=10)
        tk.Label(out_frame, text="저장 폴더:").pack(side="left")
        self.out_var = tk.StringVar(value=DEFAULT_SAVE_DIR)
        tk.Entry(out_frame, textvariable=self.out_var, width=45).pack(side="left", padx=4)
        tk.Button(out_frame, text="찾기", command=self._pick_out_dir).pack(side="left")

        name_frame = tk.Frame(self, pady=2)
        name_frame.pack(fill="x", padx=10)
        tk.Label(name_frame, text="PDF 파일명 (이미지 모드):").pack(side="left")
        self.pdf_name_var = tk.StringVar(value="output")
        tk.Entry(name_frame, textvariable=self.pdf_name_var, width=24).pack(side="left", padx=4)
        tk.Label(name_frame, text=".pdf").pack(side="left")

        upscale_frame = tk.LabelFrame(self, text="업스케일 (Real-ESRGAN)", pady=6)
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
        self.model_cb = ttk.Combobox(row1, textvariable=self.model_var,
                                     values=ESRGAN_MODELS, width=28, state="readonly")
        self.model_cb.pack(side="left", padx=4)

        row2 = tk.Frame(upscale_frame)
        row2.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(row2, text="타일:").pack(side="left", padx=(0, 2))
        self.tile_var = tk.StringVar(value="64")
        self.tile_cb = ttk.Combobox(row2, textvariable=self.tile_var,
                                    values=["0", "400", "200", "100", "64"], width=5, state="readonly")
        self.tile_cb.pack(side="left")
        tk.Label(row2, text="(깨지면 낮추기, 0=자동)", fg="gray").pack(side="left", padx=2)

        row3 = tk.Frame(upscale_frame)
        row3.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(row3, text="GPU:").pack(side="left", padx=(0, 2))
        if self._gpus:
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

        exe_status = "감지됨" if ESRGAN_EXE.exists() else "exe 없음 -- 프로그램과 같은 폴더에 넣어주세요"
        tk.Label(upscale_frame, text=exe_status,
                 fg="green" if ESRGAN_EXE.exists() else "red").pack(anchor="w", padx=4)

        # ── 진행도 ──────────────────────────────────
        prog_frame = tk.Frame(self, pady=4)
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
        self.convert_btn = tk.Button(self, text="PDF 변환 시작", font=("", 12, "bold"),
                                     bg="#2563eb", fg="white", activebackground="#1d4ed8",
                                     pady=6, command=self._convert)
        self.convert_btn.pack(fill="x", padx=10, pady=(2, 10))

        self._items = {}

    def _toggle_upscale(self):
        state = "readonly" if self.upscale_var.get() else "disabled"
        self.scale_cb.config(state=state)
        self.model_cb.config(state=state)
        self.tile_cb.config(state=state)
        self.gpu_cb.config(state=state)

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

        if self.upscale_var.get():
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
                    msg_text = "\n".join(results + errors)
                    if errors and not results:
                        messagebox.showerror("변환 실패", msg_text)
                    elif errors:
                        messagebox.showwarning("일부 실패", msg_text)
                    else:
                        messagebox.showinfo("완료!", msg_text)
                    return
        except queue.Empty:
            pass
        self.after(100, lambda: self._poll_queue(total))


if __name__ == "__main__":
    App().mainloop()
