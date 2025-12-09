import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from scipy.signal import medfilt
import numpy as np
import traceback
from PlotFrame import PlotFrame
from AnalysisPanel import AnalysisPanel

try:
    from HspyPrep import HspyPrep
    from SpecVision import CondAns
except ImportError:
    try:
        messagebox.showerror(
            "Error",
            "Could not import 'HspyPrep' or 'new_cd'.\n"
            "Make sure 'HspyPrep.py' and 'new_cd.py' are in the same directory as this app."
        )
    except Exception:
        print("ImportError: Could not import HspyPrep/new_cd")
    raise


class PreprocessingPanel:
    def __init__(self, parent, raw_dict_of_files, ref_key, bg_file_path, load_mode):
        self.parent = parent
        self.raw_data_dict = raw_dict_of_files
        self.ref_key = ref_key
        self.bg_file_path = bg_file_path
        self.load_mode = load_mode
        self.processed = False
        self.raw_spectrum = None
        self.wavelengths = None

        self.window = tk.Toplevel(parent)
        self.window.title("CL/PL Analyzer - Step 2: Preprocessing")
        self.window.geometry("980x760")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="1. Select Sample Pixel for Preview");
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        self.sample_exp_var = tk.StringVar(value=ref_key)
        ttk.Label(controls_frame, text="Experiment:").pack(side=tk.LEFT, padx=5)
        ttk.OptionMenu(controls_frame, self.sample_exp_var, ref_key, *raw_dict_of_files.keys()).pack(side=tk.LEFT,
                                                                                                     padx=5)
        ttk.Label(controls_frame, text="Row:").pack(side=tk.LEFT, padx=5)
        self.row_var = tk.IntVar(value=10)
        ttk.Entry(controls_frame, textvariable=self.row_var, width=6).pack(side=tk.LEFT)
        ttk.Label(controls_frame, text="Col:").pack(side=tk.LEFT, padx=5)
        self.col_var = tk.IntVar(value=10)
        ttk.Entry(controls_frame, textvariable=self.col_var, width=6).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Load Sample Spectrum", command=self.load_sample_plot).pack(side=tk.RIGHT,
                                                                                                    padx=10)

        plot_frame = ttk.LabelFrame(main_frame, text="Preview")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.preview_plot = PlotFrame(plot_frame, figsize=(8, 4))
        self.preview_plot.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, padx=5, pady=5)
        settings_frame = ttk.LabelFrame(bottom_frame, text="2. Preprocessing Settings");
        settings_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.remove_bg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Remove Background", variable=self.remove_bg_var,
                        command=self.update_sample_plot).pack(anchor="w", padx=10, pady=5)
        ttk.Label(settings_frame, text="Smoothing Kernel Size (must be odd):").pack(anchor="w", padx=10)
        self.kernel_var = tk.IntVar(value=17)
        ttk.Scale(settings_frame, from_=1, to=51, orient="horizontal", variable=self.kernel_var,
                  command=self.on_slider_change).pack(fill=tk.X, padx=10, pady=5)
        self.kernel_label = ttk.Label(settings_frame, text="17")
        self.kernel_label.pack(padx=10)

        action_frame = ttk.LabelFrame(bottom_frame, text="3. Apply")
        action_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.apply_all_btn = ttk.Button(action_frame, text="APPLY TO ALL", style="Accent.TButton",
                                        command=self.apply_to_all)
        self.apply_all_btn.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.proceed_btn = ttk.Button(action_frame, text="Proceed to Analysis →", command=self.open_analysis_panel,
                                      state="disabled")
        self.proceed_btn.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.load_sample_plot()

    def on_slider_change(self, value_str):
        value = int(float(value_str))
        if value % 2 == 0: value += 1
        self.kernel_var.set(value)
        self.kernel_label.config(text=str(value))
        self.update_sample_plot()

    def load_sample_plot(self):
        try:
            exp_name = self.sample_exp_var.get()
            hsp_obj = self.raw_data_dict[exp_name]
            row, col = self.row_var.get(), self.col_var.get()
            self.raw_spectrum = hsp_obj.get_numpy_spectra()[row, col, :].copy();
            self.wavelengths = hsp_obj.get_wavelengths()
            ax = self.preview_plot.ax
            ax.clear()
            ax.plot(self.wavelengths, self.raw_spectrum, label=f"Raw @ ({row},{col})")
            self.update_sample_plot()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not load sample: {e}\n{traceback.format_exc()}")

    def update_sample_plot(self):
        if self.raw_spectrum is None: return
        ax = self.preview_plot.ax
        while len(ax.lines) > 1: ax.lines[-1].remove()
        try:
            do_bg = self.remove_bg_var.get()
            kernel = self.kernel_var.get()
            preview_spectrum = self.raw_spectrum.copy()
            if do_bg:
                bg_data = self._get_background_for_preview()
                if bg_data is not None: preview_spectrum -= bg_data
            processed_spectrum = medfilt(preview_spectrum, kernel_size=kernel)
            ax.plot(self.wavelengths, processed_spectrum, label=f"Preview (BG: {do_bg}, K: {kernel})", linestyle='--')
            ax.legend(fontsize="small")
            self.preview_plot.canvas.draw()
        except Exception as e:
            print(f"Preview Error: {e}")

    def _get_background_for_preview(self):
        try:
            if self.load_mode == 'single' and self.bg_file_path:
                with open(self.bg_file_path, 'r') as f:
                    l = [line.split() for line in f]
                    background = np.array(l, dtype=float)[:, ::-1]
                    return background[1]
            elif self.load_mode == 'series':
                exp_name = self.sample_exp_var.get();
                hsp_obj = self.raw_data_dict[exp_name]
                if getattr(hsp_obj, 'background', None) is not None: return hsp_obj.background[1]
        except Exception as e:
            print(f"Error loading preview background: {e}")
        return None

    def apply_to_all(self):
        try:
            do_bg = self.remove_bg_var.get()
            kernel = self.kernel_var.get()
            self.window.config(cursor="wait")
            for exp_name, hsp_obj in self.raw_data_dict.items():
                if do_bg:
                    if self.load_mode == 'single' and self.bg_file_path:
                        hsp_obj.remove_background(file_path=self.bg_file_path)
                    elif self.load_mode == 'series':
                        hsp_obj.remove_background()
                hsp_obj.apply_filter_noises(kernel_size=kernel)
            self.processed = True
            self.apply_all_btn.config(text="Applied!", state="disabled")
            self.proceed_btn.config(state="normal")
            self.window.config(cursor="")
            messagebox.showinfo("Success", f"Preprocessing applied to all {len(self.raw_data_dict)} experiments.")
        except Exception as e:
            self.window.config(cursor="")
            messagebox.showerror("Processing Error", f"An error occurred: {e}\n{traceback.format_exc()}")

    def open_analysis_panel(self):
        if not self.processed:
            messagebox.showerror("Error", "Please apply preprocessing settings first.");
            return
        try:
            self.window.withdraw()
            cond_ans = CondAns(data_dict=self.raw_data_dict, ref=self.ref_key, addr_file="", load_mapping=False)
            AnalysisPanel(self.parent, cond_ans, self.ref_key)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize CondAns: {e}\n{traceback.format_exc()}")
            self.window.deiconify()

    def on_close(self):
        self.window.destroy()
        self.parent.deiconify()
