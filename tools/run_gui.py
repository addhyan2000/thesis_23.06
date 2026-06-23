"""
run_gui.py — Initial delivery verification panel (GPU project)

Launch:
  .\\.venv\\Scripts\\python.exe tools\\run_gui.py
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SETTINGS_PATH = PROJECT_ROOT / "gui_settings.json"

DEFAULT_EXCEL = PROJECT_ROOT / "DATASETS" / "CASME II" / "CASME2-coding-20140508.xlsx"
DEFAULT_MEDIA_AVI = PROJECT_ROOT / "Processed_Data" / "Raw_Videos_Magnified" / "CASME2"
DEFAULT_MEDIA_IMAGES = PROJECT_ROOT / "DATASETS" / "CASME II" / "Cropped"

# (status_key, short_label, full_label)
VERIFICATION_CHECKS = [
    ("env", "GPU env", "GPU environment (CUDA + PyTorch)"),
    ("smoke1", "Smoke S1", "Smoke test — Step 1 wiring"),
    ("smoke2", "Smoke S2", "Smoke test — Step 2 wiring"),
    ("smoke_ablation", "Smoke abl.", "Smoke test — ablation wiring"),
    ("preprocess", "Preprocess", "Real data — Step 1 + Step 2 (CPU)"),
    ("gpu_grouped", "GPU 3-cls", "GPU ablation — grouped labels (4 configs)"),
    ("gpu_individual", "GPU indiv.", "GPU ablation — individual emotions"),
    ("plots", "Plots", "Result plots"),
    ("literature", "Literature", "Literature comparison table"),
]

KEY_CONFIGS = [
    "config_1_pure_base",
    "config_3_spatial_only",
    "config_7_full_no_attention",
    "config_8_proposed_unified",
]


class MerTestGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("MER — Initial Delivery Test Panel (GPU)")
        self.minsize(960, 640)

        self._log_queue: queue.Queue = queue.Queue()
        self._running = False
        self._status_vars: dict[str, tk.StringVar] = {}
        self._main_paned: ttk.Panedwindow | None = None

        self.excel_var = tk.StringVar()
        self.media_var = tk.StringVar()
        self.media_mode_var = tk.StringVar(value="avi")
        self.epochs_var = tk.StringVar(value="5")
        self.workers_var = tk.StringVar(value="8")
        self.skip_preprocess_var = tk.BooleanVar(value=False)

        self._load_settings()
        self._build_ui()
        self._start_maximized()
        self.after(100, self._poll_log_queue)
        self.after_idle(self._prefer_log_space)
        self.after_idle(self._warn_if_outdated_gui)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_maximized(self) -> None:
        try:
            self.state("zoomed")
        except tk.TclError:
            try:
                self.attributes("-zoomed", True)
            except tk.TclError:
                w = self.winfo_screenwidth()
                h = self.winfo_screenheight()
                self.geometry(f"{w}x{h}+0+0")

    def _prefer_log_space(self) -> None:
        if self._main_paned is None:
            return
        try:
            total = self._main_paned.winfo_height()
            if total > 200:
                self._main_paned.sashpos(0, min(260, total // 4))
        except tk.TclError:
            pass

    def _load_settings(self) -> None:
        excel = str(DEFAULT_EXCEL)
        media = str(DEFAULT_MEDIA_AVI)
        media_mode = "avi"
        if SETTINGS_PATH.exists():
            try:
                data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                excel = data.get("casme2_excel", data.get("casme2_excel_path", excel))
                media = data.get(
                    "casme2_media_root",
                    data.get("casme2_frames_root", media),
                )
                media_mode = data.get("casme2_media_mode", media_mode)
            except (json.JSONDecodeError, OSError):
                pass
        self.excel_var.set(excel)
        self.media_var.set(media)
        self.media_mode_var.set(media_mode)

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.write_text(
                json.dumps(
                    {
                        "casme2_excel": self.excel_var.get().strip(),
                        "casme2_media_root": self.media_var.get().strip(),
                        "casme2_media_mode": self.media_mode_var.get().strip(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _warn_if_outdated_gui(self) -> None:
        """Detect an old run_gui.py copy missing recent verification features."""
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError:
            return
        markers = ("metadata CSV (fresh)", "RESULT: PASSED", "_verify_step_output")
        if all(marker in source for marker in markers):
            return
        self._append_log(
            "\nWARNING: This run_gui.py looks outdated.\n"
            "Copy the latest tools/run_gui.py from MER_Client_GPU to S:\\code\\recognition\\tools\\\n"
            "Then restart the GUI before running Preprocess.\n"
        )
        messagebox.showwarning(
            "Outdated GUI",
            "This run_gui.py is missing recent fixes.\n\n"
            "Copy the latest tools/run_gui.py from MER_Client_GPU,\n"
            "restart the GUI, then run Preprocess again.",
        )

    def _on_close(self) -> None:
        self._save_settings()
        self.destroy()

    def _build_ui(self) -> None:
        self._main_paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        self._main_paned.pack(fill="both", expand=True, padx=4, pady=4)

        controls = ttk.Frame(self._main_paned)
        log_outer = ttk.Frame(self._main_paned)
        self._main_paned.add(controls, weight=0)
        self._main_paned.add(log_outer, weight=1)

        pad = {"padx": 8, "pady": 3}

        intro = ttk.LabelFrame(controls, text="How to test")
        intro.pack(fill="x", **pad)
        ttk.Label(
            intro,
            text=(
                "Smoke tests — no CASME-II files needed (they reset Step 1 to 1 synthetic clip).\n"
                "Real-data — Excel + media root; Preprocess always rebuilds CSV + tensors (--force).\n"
                "Media type: '.avi clips' OR 'image folders' (Cropped: subXX/Filename/*.jpg)."
            ),
            justify="left",
        ).pack(anchor="w", padx=8, pady=4)

        dataset = ttk.LabelFrame(controls, text="Real-data paths (not needed for smoke tests)")
        dataset.pack(fill="x", **pad)

        self._path_row(dataset, "Excel (.xlsx):", self.excel_var, self._browse_excel, 0)
        self._path_row(dataset, "Media root:", self.media_var, self._browse_media, 1)
        mode_row = ttk.Frame(dataset)
        mode_row.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=2)
        ttk.Label(mode_row, text="Media type:").pack(side="left")
        ttk.Combobox(
            mode_row,
            textvariable=self.media_mode_var,
            values=("avi", "images"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=6)
        ttk.Label(
            mode_row,
            text="avi = subXX/clip.avi  |  images = subXX/clip/*.jpg",
            foreground="#555",
        ).pack(side="left", padx=8)

        opts = ttk.LabelFrame(controls, text="Options")
        opts.pack(fill="x", **pad)
        row0 = ttk.Frame(opts)
        row0.pack(fill="x", padx=8, pady=4)
        ttk.Label(row0, text="Ablation epochs:").pack(side="left")
        ttk.Entry(row0, textvariable=self.epochs_var, width=6).pack(side="left", padx=(4, 16))
        ttk.Label(row0, text="Step 2 workers:").pack(side="left")
        ttk.Entry(row0, textvariable=self.workers_var, width=6).pack(side="left", padx=(4, 16))
        ttk.Checkbutton(
            row0,
            text="Skip preprocessing (tensors already built)",
            variable=self.skip_preprocess_var,
        ).pack(side="left")

        btns = ttk.LabelFrame(controls, text="Run tests")
        btns.pack(fill="x", **pad)
        btn_row = ttk.Frame(btns)
        btn_row.pack(fill="x", padx=4, pady=4)
        for text, cmd in [
            ("Environment", self.run_env),
            ("Smoke (no data)", self.run_smoke_all),
            ("Preprocess", self.run_preprocess),
            ("GPU grouped", self.run_gpu_grouped),
            ("GPU individual", self.run_gpu_individual),
            ("Plots", self.run_plots),
            ("Literature", self.run_literature),
            ("▶ RUN ALL", self.run_all_tests),
        ]:
            ttk.Button(btn_row, text=text, command=cmd).pack(side="left", padx=2, pady=2)

        checklist = ttk.LabelFrame(controls, text="Verification checklist")
        checklist.pack(fill="x", **pad)
        inner = ttk.Frame(checklist)
        inner.pack(fill="x", padx=4, pady=2)
        for key, short, _full in VERIFICATION_CHECKS:
            var = tk.StringVar(value="⬜ " + short)
            self._status_vars[key] = var
            ttk.Label(inner, textvariable=var, font=("Segoe UI", 9)).pack(side="left", padx=4)

        log_frame = ttk.LabelFrame(log_outer, text="Log — drag divider above to resize")
        log_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=32, state="disabled", font=("Consolas", 10), wrap="word"
        )
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        self.status_bar = ttk.Label(self, text=f"Project: {PROJECT_ROOT}", relief="sunken")
        self.status_bar.pack(fill="x", side="bottom")

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        browse_cmd,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(parent, text="Browse…", command=browse_cmd).grid(row=row, column=2, padx=8, pady=3)
        parent.columnconfigure(1, weight=1)

    def _browse_excel(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CASME-II Excel file",
            filetypes=[("Excel", "*.xlsx"), ("All files", "*.*")],
            initialdir=str(Path(self.excel_var.get()).parent if self.excel_var.get() else PROJECT_ROOT),
        )
        if path:
            self.excel_var.set(path)
            self._save_settings()

    def _browse_media(self) -> None:
        mode = self.media_mode_var.get()
        title = (
            "Select Cropped image root (sub01…sub26 folders with jpg inside)"
            if mode == "images"
            else "Select .avi media root (sub01…sub26 with .avi files)"
        )
        path = filedialog.askdirectory(
            title=title,
            initialdir=self.media_var.get() or str(PROJECT_ROOT),
        )
        if path:
            self.media_var.set(path)
            self._save_settings()

    def _step1_argv(self, *, force: bool = False) -> list[str]:
        argv = [PYTHON, self._stage("main_step1.py"), "--dataset_mode", "casme2_only"]
        excel = self.excel_var.get().strip()
        media = self.media_var.get().strip()
        if excel:
            argv.extend(["--casme2_excel", excel])
        if media:
            argv.extend(["--casme2_frames_root", media])
        mode = self.media_mode_var.get().strip()
        if mode in ("avi", "images"):
            argv.extend(["--casme2_media_mode", mode])
        if force:
            argv.append("--force")
        return argv

    def _preprocess_steps(self, status_final: str = "preprocess") -> list[tuple[str, list[str], str | None]]:
        workers = self.workers_var.get().strip() or "8"
        return [
            ("Step 1 — metadata CSV (fresh)", self._step1_argv(force=True), status_final),
            (
                "Step 2 — EVM tensors",
                [
                    PYTHON, self._stage("main_step2.py"),
                    "--force",
                    "--max_workers", workers,
                    "--output_subdir", "tensors",
                    "--dataset_filter", "CASME_II",
                    "--expression_filter", "micro-expression",
                ],
                status_final,
            ),
            (
                "Step 2 — raw tensors",
                [
                    PYTHON, self._stage("main_step2.py"),
                    "--force",
                    "--max_workers", workers,
                    "--output_subdir", "tensors_raw",
                    "--dataset_filter", "CASME_II",
                    "--expression_filter", "micro-expression",
                ],
                status_final,
            ),
        ]

    @staticmethod
    def _step2_tensor_stats(csv_path: Path, tensor_dir: Path) -> tuple[int, int, int]:
        """Return (csv_rows, valid_clips, matching_tensor_files)."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        csv_rows = len(df)
        if "Frames_Exist" in df.columns:
            exist = df["Frames_Exist"].astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes", "y"}
            )
        else:
            exist = pd.Series(True, index=df.index)
        if "Sequence_Length" in df.columns:
            valid_df = df[exist & (df["Sequence_Length"] > 2)]
        else:
            valid_df = df[exist]
        matched = 0
        if tensor_dir.is_dir():
            for _, row in valid_df.iterrows():
                name = f"{row['Dataset']}_{row['Video_ID']}.npy"
                if (tensor_dir / name).is_file():
                    matched += 1
        return csv_rows, len(valid_df), matched

    def _real_data_ready(self) -> tuple[bool, str]:
        excel = Path(self.excel_var.get().strip())
        media = Path(self.media_var.get().strip())
        if not excel.is_file():
            return False, f"Excel file not found:\n{excel}"
        if not media.is_dir():
            return False, f"Media root folder not found:\n{media}"
        mode = self.media_mode_var.get()
        if mode == "images":
            if any(media.rglob("sub*/*/*.jpg")) or any(media.rglob("sub*/*.jpg")):
                return True, ""
            return (
                False,
                "No JPG images found under media root.\n\n"
                f"Folder: {media}\n"
                "Expected: sub01/clip_name/img0046.jpg\n"
                "(Use media type 'avi' if you use .avi clips instead.)",
            )
        avi_count = len(list(media.rglob("*.avi"))) + len(list(media.rglob("*.AVI")))
        if avi_count > 0:
            return True, ""
        return (
            False,
            "No .avi files found under media root.\n\n"
            f"Folder: {media}\n"
            "Expected: sub01/clip_name.avi",
        )

    def _has_processed_tensors(self) -> bool:
        csv = PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"
        tensors = PROJECT_ROOT / "Processed_Data" / "tensors"
        return csv.exists() and tensors.exists() and any(tensors.glob("*.npy"))

    def _verify_step_output(self, header: str) -> tuple[int, str]:
        """Post-run checks so stale smoke cache cannot look like success."""
        if "Step 1" in header:
            excel = Path(self.excel_var.get().strip())
            csv_path = PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"
            if not excel.is_file() or not csv_path.is_file():
                return 2, "\nFAILED verification: Excel or master CSV missing after Step 1.\n"
            try:
                import pandas as pd

                excel_rows = len(pd.read_excel(excel, engine="openpyxl"))
                csv_rows = len(pd.read_csv(csv_path))
            except Exception as exc:
                return 2, f"\nFAILED verification: could not compare Excel vs CSV ({exc}).\n"
            if csv_rows != excel_rows:
                return (
                    2,
                    "\nFAILED verification: Step 1 cache is stale.\n"
                    f"  Excel rows : {excel_rows}  ({excel})\n"
                    f"  CSV rows   : {csv_rows}  ({csv_path})\n"
                    "  Fix: copy updated tools/run_gui.py, then run Preprocess again.\n"
                    "  Or run Step 1 manually with --force.\n",
                )
            return 0, ""

        if "Step 2" in header:
            subdir = "tensors_raw" if "raw" in header.lower() else "tensors"
            csv_path = PROJECT_ROOT / "Processed_Data" / "master_thesis_labels.csv"
            tensor_dir = PROJECT_ROOT / "Processed_Data" / subdir
            if not csv_path.is_file():
                return 2, "\nFAILED verification: master CSV missing before Step 2.\n"
            try:
                csv_rows, valid_clips, matched = self._step2_tensor_stats(csv_path, tensor_dir)
            except Exception as exc:
                return 2, f"\nFAILED verification: could not read master CSV ({exc}).\n"
            if matched == 0:
                return (
                    2,
                    f"\nFAILED verification: no tensors in {tensor_dir}.\n"
                    "  Preprocess did not build training data.\n",
                )
            if matched < valid_clips:
                return (
                    2,
                    "\nFAILED verification: incomplete tensor set.\n"
                    f"  CSV rows          : {csv_rows}\n"
                    f"  Clips with images : {valid_clips}\n"
                    f"  Tensor files      : {matched} in {tensor_dir}\n"
                    "  If you copied video tensors, ensure every file is named\n"
                    "  CASME_II_{Video_ID}.npy and matches the Excel rows.\n"
                    "  Or tick 'Skip preprocessing' and run GPU if tensors are ready.\n",
                )
            return 0, ""

        return 0, ""

    def _set_status(self, key: str, state: str) -> None:
        short = next((s for k, s, _ in VERIFICATION_CHECKS if k == key), key)
        icons = {
            "pass": f"✅ {short}",
            "fail": f"❌ {short}",
            "run": f"🔄 {short}",
            "skip": f"⏭ {short}",
            "pending": f"⬜ {short}",
        }
        if key in self._status_vars:
            self._status_vars[key].set(icons.get(state, state))

    def _status_state(self, key: str) -> str:
        """Return pass | fail | run | skip | pending for a checklist key."""
        if key not in self._status_vars:
            return "pending"
        label = self._status_vars[key].get()
        if label.startswith("✅"):
            return "pass"
        if label.startswith("❌"):
            return "fail"
        if label.startswith("🔄"):
            return "run"
        if label.startswith("⏭"):
            return "skip"
        return "pending"

    def _finalize_checklist(self, checklist_keys: list[str], ok: bool) -> None:
        """Sync checklist icons with the overall run result."""
        unique_keys = list(dict.fromkeys(checklist_keys))
        if not unique_keys:
            return

        if len(unique_keys) == 1:
            self._set_status(unique_keys[0], "pass" if ok else "fail")
            return

        if ok:
            for key in unique_keys:
                if self._status_state(key) in ("run", "pending"):
                    self._set_status(key, "pass")
            return

        for key in unique_keys:
            state = self._status_state(key)
            if state == "run":
                self._set_status(key, "fail")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _format_run_result(msg: object) -> str:
        """Build a clear PASSED / FAILED banner for the end of each test run."""
        if isinstance(msg, dict):
            title = str(msg.get("title", "Test run"))
            ok = bool(msg.get("ok"))
            failed_step = msg.get("failed_step")
        else:
            title = "Test run"
            ok = str(msg).lower() == "ok"
            failed_step = None

        line = "=" * 60
        if ok:
            return f"\n{line}\nRESULT: PASSED — {title}\n{line}\n"
        detail = f"\n  Failed step: {failed_step}" if failed_step else ""
        return f"\n{line}\nRESULT: FAILED — {title}{detail}\n{line}\n"

    def _poll_log_queue(self) -> None:
        while True:
            try:
                item = self._log_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                kind, msg = item
                if kind == "status":
                    key, state = msg.split(":", 1)
                    self._set_status(key, state)
                elif kind == "log":
                    self._append_log(msg)
                elif kind == "done":
                    self._running = False
                    if isinstance(msg, dict):
                        self._finalize_checklist(
                            msg.get("checklist_keys", []),
                            bool(msg.get("ok")),
                        )
                    self._append_log(self._format_run_result(msg))
            else:
                self._append_log(str(item))
        self.after(100, self._poll_log_queue)

    def _guard_busy(self) -> bool:
        if self._running:
            messagebox.showwarning("Busy", "A test is already running.")
            return True
        self._save_settings()
        return False

    def _run_script_async(self, title: str, steps: list[tuple[str, list[str], str | None]]) -> None:
        if self._guard_busy():
            return
        self._running = True
        self._append_log(f"\n{'=' * 60}\n{title}\n{'=' * 60}\n")

        def worker() -> None:
            all_ok = True
            failed_step: str | None = None
            checklist_keys: list[str] = []
            for header, argv, status_key in steps:
                if status_key:
                    checklist_keys.append(status_key)
                self._log_queue.put(("log", f"\n>>> {header}\n"))
                if status_key:
                    self._log_queue.put(("status", f"{status_key}:run"))
                code = self._run_process(argv)
                if code == 0:
                    verify_code, verify_msg = self._verify_step_output(header)
                    if verify_code != 0:
                        code = verify_code
                        if verify_msg:
                            self._log_queue.put(("log", verify_msg))
                if code != 0:
                    all_ok = False
                    failed_step = header
                    if status_key:
                        self._log_queue.put(("status", f"{status_key}:fail"))
                    self._log_queue.put(("log", f"\nFAILED (exit {code}): {header}\n"))
                    break
                if status_key:
                    self._log_queue.put(("status", f"{status_key}:pass"))
            self._log_queue.put((
                "done",
                {
                    "title": title,
                    "ok": all_ok,
                    "failed_step": failed_step,
                    "checklist_keys": checklist_keys,
                },
            ))

        threading.Thread(target=worker, daemon=True).start()

    def _run_process(self, argv: list[str]) -> int:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self._log_queue.put(("log", line))
            proc.wait()
            return proc.returncode if proc.returncode is not None else 1
        except Exception as exc:
            self._log_queue.put(("log", f"ERROR: {exc}\n"))
            return 1

    def _tool(self, name: str) -> str:
        return str(PROJECT_ROOT / "tools" / name)

    def _stage(self, name: str) -> str:
        return str(PROJECT_ROOT / "Stage1_DataPipeline" / name)

    def run_env(self) -> None:
        self._run_script_async(
            "Environment check",
            [("check_environment.py", [PYTHON, self._tool("check_environment.py")], "env")],
        )

    def run_smoke_all(self) -> None:
        self._run_script_async(
            "Smoke tests (no dataset)",
            [
                ("smoke_step1_cpu.py", [PYTHON, self._tool("smoke_step1_cpu.py")], "smoke1"),
                ("smoke_step2_cpu.py", [PYTHON, self._tool("smoke_step2_cpu.py")], "smoke2"),
                ("smoke_ablation_cpu.py", [PYTHON, self._tool("smoke_ablation_cpu.py")], "smoke_ablation"),
            ],
        )

    def run_preprocess(self) -> None:
        ready, reason = self._real_data_ready()
        if not ready:
            messagebox.showerror("Real-data paths", reason)
            return
        self._run_script_async("Real-data preprocessing (CPU)", self._preprocess_steps())

    def run_gpu_grouped(self) -> None:
        epochs = self.epochs_var.get().strip() or "5"
        self._run_script_async(
            "GPU ablation — grouped labels",
            [(
                "run_ablation_gpu.py",
                [PYTHON, self._tool("run_ablation_gpu.py"), "--label_mode", "grouped",
                 "--epochs", epochs, "--configs", *KEY_CONFIGS],
                "gpu_grouped",
            )],
        )

    def run_gpu_individual(self) -> None:
        epochs = self.epochs_var.get().strip() or "5"
        self._run_script_async(
            "GPU ablation — individual labels",
            [(
                "run_ablation_gpu.py",
                [PYTHON, self._tool("run_ablation_gpu.py"), "--label_mode", "individual",
                 "--epochs", epochs, "--configs", "config_1_pure_base", "config_8_proposed_unified",
                 "--output_root", str(PROJECT_ROOT / "Ablation_Study" / "results_individual")],
                "gpu_individual",
            )],
        )

    def run_plots(self) -> None:
        self._run_script_async(
            "Generate plots",
            [("plot_ablation_results.py", [PYTHON, self._tool("plot_ablation_results.py")], "plots")],
        )

    def run_literature(self) -> None:
        self._run_script_async(
            "Literature comparison",
            [("compare_with_literature.py", [PYTHON, self._tool("compare_with_literature.py")], "literature")],
        )

    def run_all_tests(self) -> None:
        if self._guard_busy():
            return

        epochs = self.epochs_var.get().strip() or "5"
        skip_pre = self.skip_preprocess_var.get()

        steps: list[tuple[str, list[str], str | None]] = [
            ("check_environment.py", [PYTHON, self._tool("check_environment.py")], "env"),
            ("smoke_step1_cpu.py", [PYTHON, self._tool("smoke_step1_cpu.py")], "smoke1"),
            ("smoke_step2_cpu.py", [PYTHON, self._tool("smoke_step2_cpu.py")], "smoke2"),
            ("smoke_ablation_cpu.py", [PYTHON, self._tool("smoke_ablation_cpu.py")], "smoke_ablation"),
        ]

        if skip_pre or self._has_processed_tensors():
            self._set_status("preprocess", "skip")
            steps.append(("Skip preprocessing", [PYTHON, "-c", "print('Preprocessing skipped.')"], None))
        elif self._real_data_ready()[0]:
            steps.extend(self._preprocess_steps())
        else:
            self._set_status("preprocess", "skip")
            steps.append((
                "Skip preprocessing (set Excel + .avi media root, or check Skip)",
                [PYTHON, "-c", "print('Real-data paths not ready.')"],
                None,
            ))

        steps.extend([
            ("GPU grouped ablation", [
                PYTHON, self._tool("run_ablation_gpu.py"),
                "--label_mode", "grouped", "--epochs", epochs, "--configs", *KEY_CONFIGS,
            ], "gpu_grouped"),
            ("GPU individual ablation", [
                PYTHON, self._tool("run_ablation_gpu.py"),
                "--label_mode", "individual", "--epochs", epochs,
                "--configs", "config_1_pure_base", "config_8_proposed_unified",
                "--output_root", str(PROJECT_ROOT / "Ablation_Study" / "results_individual"),
            ], "gpu_individual"),
            ("plot_ablation_results.py", [PYTHON, self._tool("plot_ablation_results.py")], "plots"),
            ("compare_with_literature.py", [PYTHON, self._tool("compare_with_literature.py")], "literature"),
        ])

        self._run_script_async("Run all verification tests", steps)


def main() -> None:
    app = MerTestGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
