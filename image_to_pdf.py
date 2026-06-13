"""
이미지 → PDF 변환기
- 이미지 파일 추가: 여러 이미지를 1개 PDF로
- 상위 폴더 선택: 하위 폴더별로 각각 PDF 생성 (저장 위치/상위폴더명/ 안에 저장)
- 이미 변환된 PDF가 있으면 건너뜀
- Real-ESRGAN 업스케일 지원 (realesrgan-ncnn-vulkan.exe 필요)
"""

import queue
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
ESRGAN_EXE  = _SCRIPT_DIR / "realesrgan-ncnn-vulkan.exe"

ESRGAN_MODELS = ["realesrgan-x4plus-anime", "realesrgan-x4plus", "realesr-animevideov3"]


def images_to_pdf(image_paths, output_path):
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    if not imgs:
        raise ValueError("이미지가 없습니다.")
    imgs[0].save(output_path, save_all=True, append_images=imgs[1:])


def _esrgan_run(src_dir, dst_dir, scale, model, tile, gpu, progress_cb, total):
    for p in dst_dir.glob("*.png"):
        p.unlink()
    cmd = [str(ESRGAN_EXE), "-i", str(src_dir), "-o", str(dst_dir),
           "-n", model, "-s", str(scale), "-f", "png", "-t", str(tile)]
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


def _outputs_black(dst_dir):
    """결과 PNG 중 하나라도 완전히 검으면(픽셀 최대값 0) 실패로 판단."""
    pngs = [p for p in dst_dir.iterdir() if p.suffix.lower() == ".png"]
    if not pngs:
        return True
    for p in pngs:
        try:
            ex = Image.open(p).convert("RGB").getextrema()
        except Exception:
            return True
        if max(hi for _, hi in ex) == 0:
            return True
    return False


def upscale_folder(src_dir, dst_dir, scale, model, tile=0, progress_cb=None, status_cb=None):
    if not ESRGAN_EXE.exists():
        raise FileNotFoundError(
            "realesrgan-ncnn-vulkan.exe 를 찾을 수 없습니다.\n"
            "exe 파일을 프로그램과 같은 폴더에 넣어주세요:\n"
            "https://github.com/xinntao/Real-ESRGAN/releases"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = len([p for p in src_dir.iterdir() if p.suffix.lower() in SUPPORTED])

    # 재시도 순서: (요청 타일) → 절반씩 줄이기(>=32) → CPU 모드
    attempts = []
    if tile > 0:
        t = tile
        while t >= 32:
            attempts.append((t, None))
            t //= 2
    else:
        attempts.append((0, None))
    attempts.append((attempts[-1][0], -1))  # 마지막: CPU 폴백

    for i, (t, gpu) in enumerate(attempts):
        if i > 0 and status_cb:
            mode = "CPU 모드" if gpu == -1 else ("타일 " + str(t))
            status_cb("결과 비정상 → 재시도 (" + mode + ")")
        _esrgan_run(src_dir, dst_dir, scale, model, t, gpu, progress_cb, total)
        if not _outputs_black(dst_dir):
            return
    raise RuntimeError("업스케일 결과가 계속 검게 나옵니다 (VRAM/GPU 문제, CPU 모드까지 실패).")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("이미지 → PDF 변환기")
        self.resizable(True, True)
        self.geometry("700x710")
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
        upscale_folder(src_dir, tmp_out, int(self.scale_var.get()), self.model_var.get(),
                       int(self.tile_var.get()),
                       lambda d, t: self._q.put(("chapter", (d, t))),
                       lambda m: self._q.put(("status", m)))
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
