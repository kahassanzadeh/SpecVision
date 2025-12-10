import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, ExponentialGaussianModel, ConstantModel

from PlotFrame import PlotFrame

try:
    from HspyPrep import HspyPrep
    from SpecVision import CondAns
except ImportError as e:
    raise


class AnalysisPanel:
    """Tabbed analysis UI that mirrors new_cd.interactive_peak_fit_manual_new inside Tkinter.

    Tabs implemented:
      • Single Spectrum (manual peak fitting embedded)

    Notes:
      - Uses CondAns.__peak_fitting_manual (same algorithm as Jupyter widget) to keep results identical.
      - Adds sample map with current (row,col) marker and rectangle-draw ROI selection.
      - Provides Fit All and Fit Selection (ROI) batch fitting with XLSX export.
    """

    def __init__(self, parent, cond_ans_object: "CondAns", ref_key: str):
        self.parent = parent
        self.cond_ans = cond_ans_object
        self.ref_key = ref_key
        self.fit_bkg = None
        self.fit_bkg_flag = None

        self.window = tk.Toplevel(parent)
        self.window.title("CL/PL Analyzer — Analysis Panel")
        self.window.geometry("1280x900")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.roi = None
        self.force_square = tk.BooleanVar(value=False)

        main = ttk.Frame(self.window, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Analysis", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, text=f"Reference: {self.ref_key}").pack(side=tk.RIGHT)

        self.nb = ttk.Notebook(main)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._build_tab_single_spectrum()
        self._build_tab_waterfall()
        self._build_tab_parameter_maps()

        if hasattr(self, 'ss_map'):
            self.ss_map.canvas.mpl_connect('button_press_event', self._on_map_click)

        # 2. Bind Arrow Keys for Navigation
        self.window.bind("<Left>", self._on_arrow_navigate)
        self.window.bind("<Right>", self._on_arrow_navigate)
        self.window.bind("<Up>", self._on_arrow_navigate)
        self.window.bind("<Down>", self._on_arrow_navigate)

        self.window.focus_set()

    def _build_tab_single_spectrum(self):
        from matplotlib.widgets import RectangleSelector

        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Single Spectrum")

        top = ttk.Frame(tab)
        top.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Dataset:").pack(side=tk.LEFT)
        self.ss_exp_var = tk.StringVar(value=self.ref_key)
        keys = list(self.cond_ans.data_dict.keys())
        if self.ref_key not in keys and keys:
            self.ss_exp_var.set(keys[0])
        exp_menu = ttk.OptionMenu(top, self.ss_exp_var, self.ss_exp_var.get(), *keys,
                                  command=lambda *_: self._on_exp_change())
        exp_menu.pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Row:").pack(side=tk.LEFT, padx=(12, 2))
        self.ss_row = tk.IntVar(value=0)
        ttk.Button(top, text="←", width=3, command=lambda: self._bump(self.ss_row, -1)).pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.ss_row, width=6).pack(side=tk.LEFT)
        ttk.Button(top, text="→", width=3, command=lambda: self._bump(self.ss_row, +1)).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(top, text="Col:").pack(side=tk.LEFT, padx=(12, 2))
        self.ss_col = tk.IntVar(value=0)
        ttk.Button(top, text="←", width=3, command=lambda: self._bump(self.ss_col, -1)).pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.ss_col, width=6).pack(side=tk.LEFT)
        ttk.Button(top, text="→", width=3, command=lambda: self._bump(self.ss_col, +1)).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(top, text="Preview", command=self._ss_preview).pack(side=tk.LEFT, padx=10)

        self.live_fit = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Live Fit", variable=self.live_fit).pack(side=tk.LEFT, padx=10)

        # ttk.Label(top, text="Medfilt k (odd):").pack(side=tk.LEFT, padx=(16, 2)) self.ss_kernel = tk.IntVar(
        # value=11) ttk.Spinbox(top, from_=1, to=99, increment=2, textvariable=self.ss_kernel, width=4,
        # command=self._ss_preview).pack(side=tk.LEFT)

        self.show_components = tk.BooleanVar(value=True)
        self.fit_bkg = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Show Components", variable=self.show_components, command=self._ss_preview).pack(
            side=tk.LEFT, padx=10)
        ttk.Checkbutton(top, text="Automatic Background Fit", variable=self.fit_bkg, command=self._check_fit_bkg).pack(
            side=tk.LEFT, padx=10)

        ttk.Button(top, text="Save Figure…", command=self._ss_save_png).pack(side=tk.RIGHT)

        # ---- Two panels: spectrum (left) + sample map (right) ----
        split = ttk.Frame(tab)
        split.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 8))

        # Left: spectrum
        left = ttk.Frame(split)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ss_plot = PlotFrame(left, figsize=(8.6, 3.8))
        self.ss_plot.pack(fill=tk.BOTH, expand=True)

        # Right: sample map (with RectangleSelector)
        right = ttk.Frame(split)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.ss_map = PlotFrame(right, figsize=(4.2, 3.8))
        self.ss_map.pack(fill=tk.BOTH, expand=True)
        self.ss_map.canvas.mpl_connect('button_press_event', self._on_map_click)
        self._rs = None

        # ---- Manual peak builder (mirrors ipywidgets UI) ----
        mf = ttk.LabelFrame(tab, text="Manual Peak Fit (Tkinter port)")
        mf.pack(fill=tk.X, padx=8, pady=(0, 8))

        ctl = ttk.Frame(mf)
        ctl.pack(fill=tk.X, padx=6, pady=6)

        ttk.Button(ctl, text="＋ Add Peak", command=self._add_peak_row).pack(side=tk.LEFT)
        ttk.Button(ctl, text="− Remove Last", command=self._remove_last_peak_row).pack(side=tk.LEFT, padx=6)

        ttk.Button(ctl, text="Import Params", command=self._import_peak_params).pack(side=tk.LEFT, padx=(20, 2))
        ttk.Button(ctl, text="Export Params", command=self._export_peak_params).pack(side=tk.LEFT)

        ttk.Label(ctl, text="R² threshold:").pack(side=tk.LEFT, padx=(16, 4))
        self.ss_r2_thresh = tk.DoubleVar(value=0.005)
        ttk.Entry(ctl, textvariable=self.ss_r2_thresh, width=8).pack(side=tk.LEFT)

        ttk.Button(ctl, text="Fit", style="Accent.TButton", command=self._ss_fit_embedded).pack(side=tk.RIGHT)
        self.ss_r2_label = ttk.Label(ctl, text="R²: –")
        self.ss_r2_label.pack(side=tk.RIGHT, padx=10)

        # Batch controls (Fit All / ROI)
        batch = ttk.Frame(mf)
        batch.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Checkbutton(batch, text="Force square (expand short side)", variable=self.force_square).pack(side=tk.LEFT)
        ttk.Label(batch, text="r start:").pack(side=tk.LEFT, padx=(12, 2))
        self.roi_r0 = tk.IntVar(value=0)
        ttk.Entry(batch, textvariable=self.roi_r0, width=6).pack(side=tk.LEFT)
        ttk.Label(batch, text="r end:").pack(side=tk.LEFT, padx=(12, 2))
        self.roi_r1 = tk.IntVar(value=0)
        ttk.Entry(batch, textvariable=self.roi_r1, width=6).pack(side=tk.LEFT)
        ttk.Label(batch, text="c start:").pack(side=tk.LEFT, padx=(12, 2))
        self.roi_c0 = tk.IntVar(value=0)
        ttk.Entry(batch, textvariable=self.roi_c0, width=6).pack(side=tk.LEFT)
        ttk.Label(batch, text="c end:").pack(side=tk.LEFT, padx=(12, 2))
        self.roi_c1 = tk.IntVar(value=0)
        ttk.Entry(batch, textvariable=self.roi_c1, width=6).pack(side=tk.LEFT)
        ttk.Button(batch, text="Select ROI (draw)", command=self._start_rect_select).pack(side=tk.LEFT, padx=(12, 6))
        ttk.Button(batch, text="Fit Selection", command=self._fit_selection).pack(side=tk.LEFT, padx=4)
        ttk.Button(batch, text="Fit All", command=self._fit_all).pack(side=tk.LEFT, padx=4)

        # Peak table (read-only results)
        table_wrap = ttk.Frame(mf)
        table_wrap.pack(fill=tk.X, padx=6, pady=(0, 6))
        cols = ("prefix", "amplitude", "center", "sigma", "fwhm")
        self.param_table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=6)
        for c in cols:
            self.param_table.heading(c, text=c)
            self.param_table.column(c, width=120, anchor=tk.CENTER)
        self.param_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb = ttk.Scrollbar(table_wrap, orient=tk.VERTICAL, command=self.param_table.yview)
        self.param_table.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        self.ss_peaks_container = ttk.Frame(mf)
        self.ss_peaks_container.pack(fill=tk.X, padx=6, pady=(6, 6))
        self.ss_peak_rows = []  # list[dict]

        self._add_peak_row(default_exists=False)

        self._ss_preview()
        self._draw_map()

    # --------------------------- helpers / actions ---------------------------
    def _bump(self, var: tk.IntVar, delta: int):
        try:
            var.set(max(0, int(var.get()) + delta))
            self._ss_preview()
            self._draw_map()

            if self.live_fit.get() and len(self.ss_peak_rows) > 0:
                self._ss_fit_embedded()

        except Exception:
            pass

    def _on_exp_change(self):
        self.roi = None
        self.ss_row.set(0)
        self.ss_col.set(0)
        self._ss_preview()
        self._draw_map()

    def _get_current_data(self):
        exp = self.ss_exp_var.get()
        hsp = self.cond_ans.data_dict[exp]
        data = hsp.get_numpy_spectra()
        wl = hsp.get_wavelengths()
        # clamp row/col
        r = int(np.clip(self.ss_row.get(), 0, data.shape[0] - 1))
        c = int(np.clip(self.ss_col.get(), 0, data.shape[1] - 1))
        return hsp, data, wl, r, c

    def _ss_preview(self):
        try:
            _, data, wl, r, c = self._get_current_data()
            spec = data[r, c, :].astype(float)
            # k = int(self.ss_kernel.get())
            # if k % 2 == 0:
            #     k += 1
            # sm = medfilt(spec, kernel_size=k)
            ax = self.ss_plot.ax
            ax.clear()
            ax.plot(wl, spec)
            # ax.plot(wl, sm, "--", label=f"Medfilt k={k}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            # ax.legend()
            self.ss_plot.canvas.draw()
        except Exception as e:
            messagebox.showerror("Single Spectrum", f"Failed to preview: {e}\n{traceback.format_exc()}")

    # ---- map + ROI drawing ----
    def _draw_map(self):
        try:
            hsp, data, _, r, c = self._get_current_data()

            img = None
            if hasattr(hsp, 'get_live_scan'):
                try:
                    img = hsp.get_live_scan()
                except Exception:
                    img = None

            ax = self.ss_map.ax
            ax.clear()

            if img is None:
                ax.text(0.5, 0.5, 'No map image\nprovided', ha='center', va='center')
                ax.axis('off')
            else:
                n_rows, n_cols = data.shape[:2]
                ih, iw = img.shape[:2]
                cw, ch = iw / n_cols, ih / n_rows
                ax.imshow(img, cmap='gray', extent=[0, iw, ih, 0])

                ax.set_title('Sample map')
                ax.set_axis_off()

                rect = plt.Rectangle((c * cw, r * ch), cw, ch, linewidth=1.6, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                roi = self._current_roi_bounds()
                if roi is not None:
                    r0, r1, c0, c1 = roi
                    rx, ry = c0 * cw, r0 * ch
                    rw, rh = (c1 - c0 + 1) * cw, (r1 - r0 + 1) * ch
                    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, linewidth=1.4, edgecolor='yellow', facecolor='none',
                                               linestyle='--'))

            self.ss_map.canvas.draw()

        except Exception as e:
            messagebox.showerror("Map", f"Failed to draw map: {e}\n{traceback.format_exc()}")

    def _start_rect_select(self):
        from matplotlib.widgets import RectangleSelector
        # activate rectangle selector on map axes
        if self._rs is not None:
            try:
                self._rs.disconnect_events();
                self._rs = None
            except Exception:
                pass
        ax = self.ss_map.ax
        n_rows, n_cols = self.cond_ans.data_dict[self.ss_exp_var.get()].get_numpy_spectra().shape[:2]
        ih, iw = None, None
        img = getattr(self.cond_ans.data_dict[self.ss_exp_var.get()], 'get_live_scan', lambda: None)()
        if img is not None:
            ih, iw = img.shape[:2]
        else:
            # fall back to a grid in axes coords
            bbox = ax.get_window_extent().transformed(self.ss_map.fig.dpi_scale_trans.inverted())
            iw, ih = bbox.width * self.ss_map.fig.dpi, bbox.height * self.ss_map.fig.dpi

        cw, ch = iw / n_cols, ih / n_rows

        def onselect(eclick, erelease):
            # pixel-space to (r0,r1,c0,c1)
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            c0, c1 = int(np.floor(xmin / cw)), int(np.floor(xmax / cw))
            r0, r1 = int(np.floor(ymin / ch)), int(np.floor(ymax / ch))
            r0 = max(0, min(r0, n_rows - 1))
            r1 = max(0, min(r1, n_rows - 1))
            c0 = max(0, min(c0, n_cols - 1))
            c1 = max(0, min(c1, n_cols - 1))
            if self.force_square.get():
                dr, dc = (r1 - r0 + 1), (c1 - c0 + 1)
                if dr > dc:
                    # expand columns
                    extra = dr - dc
                    pad_l = extra // 2
                    pad_r = extra - pad_l
                    c0 = max(0, c0 - pad_l)
                    c1 = min(n_cols - 1, c1 + pad_r)
                elif dc > dr:
                    extra = dc - dr
                    pad_t = extra // 2
                    pad_b = extra - pad_t
                    r0 = max(0, r0 - pad_t)
                    r1 = min(n_rows - 1, r1 + pad_b)
            self.roi_r0.set(r0)
            self.roi_r1.set(r1)
            self.roi_c0.set(c0)
            self.roi_c1.set(c1)
            self._draw_map()

        self._rs = RectangleSelector(ax, onselect, useblit=True, interactive=False, button=[1])

    def _current_roi_bounds(self):
        try:
            r0, r1 = int(self.roi_r0.get()), int(self.roi_r1.get())
            c0, c1 = int(self.roi_c0.get()), int(self.roi_c1.get())
            if r1 < r0 or c1 < c0:
                return None
            return r0, r1, c0, c1
        except Exception:
            return None

    def _ss_save_png(self):
        try:
            fn = filedialog.asksaveasfilename(title="Save current figure…", defaultextension=".png",
                                              filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                                                         ("All", "*.*")])
            if not fn:
                return
            # save combined fig: create a temp fig with 1x2 panels
            fig = plt.figure(figsize=(11, 4))
            ax1 = fig.add_subplot(1, 2, 1)
            for ln in self.ss_plot.ax.get_lines():
                ax1.plot(ln.get_xdata(), ln.get_ydata(), linestyle=ln.get_linestyle(), linewidth=ln.get_linewidth())
            ax1.set_xlabel("Wavelength (nm)");
            ax1.set_ylabel("Intensity (a.u.)")
            ax1.set_title("Spectrum")
            ax2 = fig.add_subplot(1, 2, 2)
            for im in self.ss_map.ax.get_images():
                ax2.imshow(im.get_array(), cmap='gray')
            for p in self.ss_map.ax.patches:
                ax2.add_patch(plt.Rectangle((p.get_x(), p.get_y()), p.get_width(), p.get_height(), fill=False,
                                            linestyle=p.get_linestyle(), linewidth=p.get_linewidth(),
                                            edgecolor=p.get_edgecolor()))
            ax2.set_axis_off()
            ax2.set_title("Sample map")
            fig.savefig(fn, dpi=300, bbox_inches="tight");
            plt.close(fig)
        except Exception as e:
            messagebox.showerror("Save Figure", f"Could not save figure: {e}")

    # ----- Dynamic peak-row builder -----
    def _add_peak_row(self, default_exists: bool = False):
        rowf = ttk.Frame(self.ss_peaks_container)
        rowf.pack(fill=tk.X, pady=3)

        # Which model
        func_var = tk.StringVar(value='GaussianModel')
        func_cb = ttk.Combobox(rowf, state='readonly', width=18, textvariable=func_var,
                               values=['GaussianModel', 'LorentzianModel', 'VoigtModel', 'ExponentialGaussianModel'])
        func_cb.pack(side=tk.LEFT, padx=4)

        # use/exist toggle (default False to match notebook)
        exists = tk.BooleanVar(value=default_exists)
        ttk.Checkbutton(rowf, text='Always Exists', variable=exists).pack(side=tk.LEFT, padx=(2, 8))

        # amplitude
        amp = tk.DoubleVar(value=1.0)
        amp_min = tk.DoubleVar(value=0.0)
        amp_max = tk.DoubleVar(value=0.0)
        ttk.Label(rowf, text="A:").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=amp, width=8).pack(side=tk.LEFT)
        ttk.Label(rowf, text="min").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=amp_min, width=6).pack(side=tk.LEFT)
        ttk.Label(rowf, text="max").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=amp_max, width=6).pack(side=tk.LEFT)

        # center
        cen = tk.DoubleVar(value=0.0)
        cen_min = tk.DoubleVar(value=0.0)
        cen_max = tk.DoubleVar(value=0.0)
        ttk.Label(rowf, text="μ:").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=cen, width=8).pack(side=tk.LEFT)
        ttk.Label(rowf, text="min").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=cen_min, width=6).pack(side=tk.LEFT)
        ttk.Label(rowf, text="max").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=cen_max, width=6).pack(side=tk.LEFT)

        # sigma
        sig = tk.DoubleVar(value=1.0)
        sig_min = tk.DoubleVar(value=0.0)
        sig_max = tk.DoubleVar(value=0.0)
        ttk.Label(rowf, text="σ:").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=sig, width=8).pack(side=tk.LEFT)
        ttk.Label(rowf, text="min").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=sig_min, width=6).pack(side=tk.LEFT)
        ttk.Label(rowf, text="max").pack(side=tk.LEFT)
        ttk.Entry(rowf, textvariable=sig_max, width=6).pack(side=tk.LEFT)

        # gamma (only ExponentialGaussianModel)
        gam = tk.DoubleVar(value=1.0)
        gam_min = tk.DoubleVar(value=0.0)
        gam_max = tk.DoubleVar(value=0.0)
        gam_wrap = ttk.Frame(rowf)
        gam_wrap.pack(side=tk.LEFT)
        e_g = ttk.Entry(gam_wrap, textvariable=gam, width=8);
        e_g.pack(side=tk.LEFT)
        ttk.Label(gam_wrap, text="min").pack(side=tk.LEFT)
        e_gmin = ttk.Entry(gam_wrap, textvariable=gam_min, width=6);
        e_gmin.pack(side=tk.LEFT)
        ttk.Label(gam_wrap, text="max").pack(side=tk.LEFT)
        e_gmax = ttk.Entry(gam_wrap, textvariable=gam_max, width=6);
        e_gmax.pack(side=tk.LEFT)

        def _toggle_gamma(*_):
            state = 'normal' if func_var.get() == 'ExponentialGaussianModel' else 'disabled'
            for w in (e_g, e_gmin, e_gmax):
                try:
                    w.configure(state=state)
                except Exception:
                    pass

        func_cb.bind('<<ComboboxSelected>>', _toggle_gamma)
        _toggle_gamma()

        self.ss_peak_rows.append({
            'frame': rowf,
            'func_widget': func_cb,
            'func': func_var,
            'amp': amp, 'amp_min': amp_min, 'amp_max': amp_max,
            'cen': cen, 'cen_min': cen_min, 'cen_max': cen_max,
            'sig': sig, 'sig_min': sig_min, 'sig_max': sig_max,
            'gam': gam, 'gam_min': gam_min, 'gam_max': gam_max,
            'exists': exists
        })

        # self.ss_peak_rows.append(dict(frame=rowf, func=func_var,
        #                               amp=amp, amp_min=amp_min, amp_max=amp_max,
        #                               cen=cen, cen_min=cen_min, cen_max=cen_max,
        #                               sig=sig, sig_min=sig_min, sig_max=sig_max,
        #                               gam=gam, gam_min=gam_min, gam_max=gam_max,
        #                               exists=exists))

    def _remove_last_peak_row(self):
        if not self.ss_peak_rows:
            return
        row = self.ss_peak_rows.pop()
        row['frame'].destroy()

    def _collect_peak_params(self):
        mapping = {
            'GaussianModel': GaussianModel,
            'LorentzianModel': LorentzianModel,
            'VoigtModel': VoigtModel,
            'ExponentialGaussianModel': ExponentialGaussianModel,
        }
        plist = []
        for r in self.ss_peak_rows:
            func_cls = mapping[r['func'].get()]
            entry = {
                'amplitude': float(r['amp'].get()),
                'amplitude_min': None if float(r['amp_min'].get()) == 0 else float(r['amp_min'].get()),
                'amplitude_max': None if float(r['amp_max'].get()) == 0 else float(r['amp_max'].get()),
                'center': float(r['cen'].get()),
                'center_min': None if float(r['cen_min'].get()) == 0 else float(r['cen_min'].get()),
                'center_max': None if float(r['cen_max'].get()) == 0 else float(r['cen_max'].get()),
                'sigma': float(r['sig'].get()),
                'sigma_min': None if float(r['sig_min'].get()) == 0 else float(r['sig_min'].get()),
                'sigma_max': None if float(r['sig_max'].get()) == 0 else float(r['sig_max'].get()),
                'func': func_cls,
                'exists': bool(r['exists'].get()),
            }
            if func_cls is ExponentialGaussianModel:
                entry.update({
                    'gamma': float(r['gam'].get()),
                    'gamma_min': None if float(r['gam_min'].get()) == 0 else float(r['gam_min'].get()),
                    'gamma_max': None if float(r['gam_max'].get()) == 0 else float(r['gam_max'].get()),
                })
            plist.append(entry)
        return plist

    def __peak_fitting_manual(self, intensity, wavelengths, peak_params, r2_threshold=0.005):
        peak_params.sort(key=lambda p: (p['exists'], p['amplitude']), reverse=True)
        if self.fit_bkg_flag:
            composite_model = ConstantModel(prefix='bkg_')
            params = composite_model.make_params(bkg_c=0)
        else:
            composite_model = None
            params = None

        self.temp = peak_params
        self.best_r2 = None
        self.best_result = None
        self.best_model = None
        self.params_fit = None
        self.best_fit = None
        always_exists_number = 0

        r2_list = []
        params_list = []
        models = []
        results = []

        def apply_bounds(par, base, min_val=None, max_val=None):
            """
            Set par.value=base; optionally set par.min and par.max.
            """
            par.set(value=base)
            if min_val is not None:
                par.min = min_val
            if max_val is not None:
                par.max = max_val

        for i, p in enumerate(peak_params):
            model = p['func'](prefix=f'p{i}_')
            composite_model = model if composite_model is None else composite_model + model

            model_params = model.make_params()
            params = model_params if params is None else params.update(model_params) or params

            apply_bounds(
                params[f'p{i}_amplitude'],
                p['amplitude'],
                p.get('amplitude_min'),
                p.get('amplitude_max')
            )
            apply_bounds(
                params[f'p{i}_center'],
                p['center'],
                p.get('center_min'),
                p.get('center_max')
            )
            apply_bounds(
                params[f'p{i}_sigma'],
                p['sigma'],
                p.get('sigma_min'),
                p.get('sigma_max')
            )
            if p['func'] is ExponentialGaussianModel:
                apply_bounds(
                    params[f'p{i}_gamma'],
                    p.get('gamma'),
                    p.get('gamma_min'),
                    p.get('gamma_max')
                )

            result = composite_model.fit(intensity, params, x=wavelengths)
            ss_total = np.sum((intensity - intensity.mean()) ** 2)
            ss_residual = np.sum(result.residual ** 2)
            r_squared = 1 - ss_residual / ss_total

            if p.get('exists') or len(r2_list) == 0:
                r2_list.append(r_squared)
                params_list.append(result.params)
                models.append(composite_model)
                results.append(result)

                # print(f"Using {i+1} peak(s): R² = {r_squared:.4f}")

                self.best_fit = i
                self.params_fit = params_list[-1]
                self.best_model = models[-1]
                self.best_result = results[-1]
                self.best_r2 = r2_list[-1]
            else:
                if r_squared - r2_list[-1] > r2_threshold:
                    r2_list.append(r_squared)
                    params_list.append(result.params)
                    models.append(composite_model)
                    results.append(result)

                    # print(f"Using {i+1} peak(s): R² = {r_squared:.4f}")

                    self.best_fit = i
                    self.params_fit = params_list[-1]
                    self.best_model = models[-1]
                    self.best_result = results[-1]
                    self.best_r2 = r2_list[-1]

    # ---------------------- fitting (Tk-embedded) ----------------------
    def _ss_fit_embedded(self):
        try:
            _, data, wl, r, c = self._get_current_data()
            intensity = data[r, c, :].astype(float)
            peaks = self._collect_peak_params()
            if not peaks:
                messagebox.showinfo("Peak Fit", "Add at least one peak.")
                return

            # Call the *same* algorithm used by the ipywidgets UI
            # if hasattr(self.cond_ans, '_CondAns__peak_fitting_manual'):
            #     fitfn = getattr(self.cond_ans, '_CondAns__peak_fitting_manual')
            # else:
            #     raise AttributeError("Private method __peak_fitting_manual not found on CondAns")

            r2_thresh = float(self.ss_r2_thresh.get())
            # fitfn(intensity=intensity, wavelengths=wl, peak_params=peaks, r2_threshold=r2_thresh)
            self.__peak_fitting_manual(intensity=intensity, wavelengths=wl, peak_params=peaks, r2_threshold=r2_thresh)
            model_res = self.best_result
            model_obj = self.best_model
            best_r2 = self.best_r2

            ax = self.ss_plot.ax
            ax.clear()
            if self.fit_bkg_flag:
                ax.plot(wl, intensity - self.params_fit['bkg_c'].value, label='data')
            else:
                ax.plot(wl, intensity, label='data')
            if model_res is not None:
                if self.fit_bkg_flag:
                    ax.plot(wl, model_res.best_fit - self.params_fit['bkg_c'].value, label='fitted data')
                else:
                    ax.plot(wl, model_res.best_fit, label='best fit')
                if self.show_components.get() and model_obj is not None:
                    comps = model_obj.eval_components(params=self.params_fit, x=wl)
                    for name, y in comps.items():
                        if name.endswith('bkg_'):
                            continue
                        ax.plot(wl, y, alpha=0.7, linestyle='-', label=name)
                self._fill_param_table(model_res.params)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend()
            self.ss_plot.canvas.draw()

            if best_r2 is not None:
                self.ss_r2_label.config(text=f"R²: {best_r2:.4f}")
            else:
                self.ss_r2_label.config(text="R²: –")

        except Exception as e:
            messagebox.showerror("Embedded Peak Fit", f"Fit failed: {e}\n{traceback.format_exc()}")

    # ---- batch fitting helpers ----
    def _flatten_params(self, params):
        # Build row dict like p0_amplitude, p0_center, ...
        row = {}
        for name, par in params.items():
            try:
                val = float(par.value)
            except Exception:
                val = par.value
            row[name] = val
        return row

    def _fit_iter(self, rows_cols, data, wl, peaks, r2_thresh):
        results = []

        prog = tk.Toplevel(self.window)
        prog.title("Fitting…")
        prog.geometry("350x150")
        prog.transient(self.window)
        prog.grab_set()

        lbl_var = tk.StringVar(value="Initializing fit...")
        ttk.Label(prog, textvariable=lbl_var).pack(pady=(15, 5))

        pb = ttk.Progressbar(prog, mode='determinate', maximum=len(rows_cols))
        pb.pack(fill=tk.X, padx=20, pady=5)

        stop_flag = [False]

        def on_cancel():
            stop_flag[0] = True
            lbl_var.set("Cancelling... please wait for current pixel.")

        btn = ttk.Button(prog, text="Stop / Cancel", command=on_cancel)
        btn.pack(pady=10)

        prog.update()

        total = len(rows_cols)
        for i, (r, c) in enumerate(rows_cols, start=1):

            if stop_flag[0]:
                break

            try:
                intensity = data[r, c, :].astype(float)

                # Perform the fit
                self.cond_ans._CondAns__peak_fitting_manual(
                    intensity=intensity,
                    wavelengths=wl,
                    peak_params=peaks,
                    r2_threshold=r2_thresh
                )

                res = getattr(self.cond_ans, 'best_result', None)
                r2 = getattr(self.cond_ans, 'best_r2', None)

                row = {"row": r, "col": c, "r2": r2}
                if res is not None:
                    row.update(self._flatten_params(res.params))
                results.append(row)

            except Exception as ex:
                results.append({"row": r, "col": c, "error": str(ex)})

            pb['value'] = i
            lbl_var.set(f"Fitting pixel {i} of {total}...")

            prog.update()

        # Cleanup
        prog.destroy()
        return results

    def _fit_selection(self):
        try:
            hsp, data, wl, _, _ = self._get_current_data()
            peaks = self._collect_peak_params()
            if not peaks:
                messagebox.showinfo("Fit Selection", "Add at least one peak first.")
                return
            roi = self._current_roi_bounds()
            if roi is None:
                messagebox.showinfo("Fit Selection", "Set ROI (r0/r1/c0/c1) or draw it on the map.")
                return
            r0, r1, c0, c1 = roi
            coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            results = self._fit_iter(coords, data, wl, peaks, float(self.ss_r2_thresh.get()))
            self._export_results(results, default_name=f"lmfit_parameters_{self.ss_exp_var.get()}_roi.xlsx")
        except Exception as e:
            messagebox.showerror("Fit Selection", f"Failed: {e}\n{traceback.format_exc()}")

    def _fit_all(self):
        try:
            hsp, data, wl, _, _ = self._get_current_data()
            peaks = self._collect_peak_params()
            if not peaks:
                messagebox.showinfo("Fit All", "Add at least one peak first.")
                return
            n_rows, n_cols = data.shape[:2]
            coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
            results = self._fit_iter(coords, data, wl, peaks, float(self.ss_r2_thresh.get()))
            self._export_results(results, default_name=f"lmfit_parameters_{self.ss_exp_var.get()}_all.xlsx")
        except Exception as e:
            messagebox.showerror("Fit All", f"Failed: {e}\n{traceback.format_exc()}")

    def _export_results(self, results, default_name="lmfit_parameters.xlsx"):
        try:
            import pandas as pd
            if not results:
                messagebox.showinfo("Export", "No results to export.")
                return
            df = pd.DataFrame(results)
            fn = filedialog.asksaveasfilename(title="Save results (Excel)", defaultextension=".xlsx",
                                              initialfile=default_name,
                                              filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All", "*.*")])
            if not fn:
                return
            if fn.lower().endswith('.csv'):
                df.to_csv(fn, index=False)
            else:
                with pd.ExcelWriter(fn, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='fits')
            messagebox.showinfo("Export", f"Saved {len(results)} rows to\n{fn}")
        except Exception as e:
            messagebox.showerror("Export", f"Failed to save results: {e}")

    def _check_fit_bkg(self):
        self.fit_bkg_flag = self.fit_bkg.get()

    def _fill_param_table(self, params):
        """Fill the read-only parameters table with lmfit Parameters."""
        try:
            self.param_table.delete(*self.param_table.get_children())
            names = list(params.keys())
            prefixes = sorted({name.split('_', 1)[0] for name in names if '_' in name})
            suffix_order = ['amplitude', 'center', 'sigma', 'fwhm']
            for pfx in prefixes:
                row = {s: '' for s in suffix_order}
                for suf in suffix_order:
                    if pfx == 'bkg':
                        row['amplitude'] = params['bkg_c'].value
                        row['center'] = '0'
                        row['sigma'] = '0'
                        row['fwhm'] = '0'
                        continue
                    key = f"{pfx}_{suf}"
                    if key in params:
                        par = params[key]
                        if getattr(par, 'stderr', None) is not None:
                            row[suf] = f"{par.value:.3f} ± {par.stderr:.3f}"
                        else:
                            row[suf] = f"{par.value:.3f}"
                self.param_table.insert('', 'end',
                                        values=(pfx, row['amplitude'], row['center'], row['sigma'], row['fwhm']))

        except Exception:
            pass

    def _ss_peakfit_call(self):
        try:
            exp = self.ss_exp_var.get()
            row, col = int(self.ss_row.get()), int(self.ss_col.get())
            if not hasattr(self.cond_ans, 'interactive_peak_fit_manual_new'):
                raise AttributeError("CondAns.interactive_peak_fit_manual_new is not available")
            self.cond_ans.interactive_peak_fit_manual_new(exp, row, col)
        except Exception as e:
            messagebox.showwarning("Peak Fit (ipywidgets)", "Tried to open the Jupyter widget version. In a Tkinter "
                                                            "app this won't display.\n\n" f"Details: {e}")

    def _on_arrow_navigate(self, event):
        """Handle keyboard arrow keys to move the pixel selection."""
        delta_r, delta_c = 0, 0

        if event.keysym == 'Up':
            delta_r = -1
        elif event.keysym == 'Down':
            delta_r = 1
        elif event.keysym == 'Left':
            delta_c = -1
        elif event.keysym == 'Right':
            delta_c = 1
        else:
            return

        try:
            hsp, data, _, _, _ = self._get_current_data()
            n_rows, n_cols = data.shape[:2]

            current_r = self.ss_row.get()
            current_c = self.ss_col.get()

            new_r = current_r + delta_r
            new_c = current_c + delta_c

            if 0 <= new_r < n_rows and 0 <= new_c < n_cols:
                self.ss_row.set(new_r)
                self.ss_col.set(new_c)

                self._ss_preview()
                self._draw_map()

                if len(self.ss_peak_rows) > 0:
                    self._ss_fit_embedded()
        except Exception:
            pass

    def _on_map_click(self, event):
        """Handle mouse clicks on the map."""
        if event.inaxes != self.ss_map.ax: return
        if self.ss_map.canvas.toolbar.mode != "": return

        try:
            hsp, data, _, _, _ = self._get_current_data()
            n_rows, n_cols = data.shape[:2]

            # Calculate indices (same logic as before)
            img = getattr(hsp, 'get_live_scan', lambda: None)()
            if img is None: return

            ih, iw = img.shape[:2]
            cw, ch = iw / n_cols, ih / n_rows

            c = int(event.xdata / cw)
            r = int(event.ydata / ch)

            if 0 <= r < n_rows and 0 <= c < n_cols:
                self.ss_row.set(r)
                self.ss_col.set(c)

                self._ss_preview()
                self._draw_map()

                if self.live_fit.get() and len(self.ss_peak_rows) > 0:
                    self._ss_fit_embedded()

        except Exception:
            pass

    def on_close(self):
        try:
            self.window.destroy()
            if self.parent is not None:
                try:
                    self.parent.deiconify()
                except Exception:
                    pass
        except Exception:
            pass

    def _build_tab_waterfall(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Waterfall Plots")

        # --- Top Controls ---
        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: Dataset & Mode
        r1 = ttk.Frame(controls)
        r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Dataset:").pack(side=tk.LEFT)
        self.wf_exp_var = tk.StringVar(value=self.ref_key)
        keys = list(self.cond_ans.data_dict.keys())
        if self.ref_key not in keys and keys: self.wf_exp_var.set(keys[0])
        om = ttk.OptionMenu(r1, self.wf_exp_var, self.wf_exp_var.get(), *keys,
                            command=lambda _: self._draw_wf_selector_map())
        om.pack(side=tk.LEFT, padx=5)

        self.wf_mode = tk.StringVar(value="Line")
        ttk.Label(r1, text="|  Shape:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Radiobutton(r1, text="Line", variable=self.wf_mode, value="Line", command=self._draw_wf_selector_map).pack(
            side=tk.LEFT)
        ttk.Radiobutton(r1, text="Rect", variable=self.wf_mode, value="Rect", command=self._draw_wf_selector_map).pack(
            side=tk.LEFT)

        # Row 2: Coordinates
        r2 = ttk.Frame(controls)
        r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Start(r,c):").pack(side=tk.LEFT)
        self.wf_r0 = tk.IntVar(value=0)
        self.wf_c0 = tk.IntVar(value=0)
        ttk.Entry(r2, textvariable=self.wf_r0, width=4).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.wf_c0, width=4).pack(side=tk.LEFT)

        ttk.Label(r2, text="  End(r,c):").pack(side=tk.LEFT)
        self.wf_r1 = tk.IntVar(value=5)
        self.wf_c1 = tk.IntVar(value=5)
        ttk.Entry(r2, textvariable=self.wf_r1, width=4).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.wf_c1, width=4).pack(side=tk.LEFT)

        ttk.Label(r2, text="| Stride:").pack(side=tk.LEFT, padx=(10, 0))
        self.wf_stride = tk.IntVar(value=1)
        ttk.Spinbox(r2, from_=1, to=20, textvariable=self.wf_stride, width=3).pack(side=tk.LEFT)

        # Row 3: Visualization & Limits
        r3 = ttk.Frame(controls)
        r3.pack(fill=tk.X, pady=2)

        # --- NEW: Wavelength Range ---
        ttk.Label(r3, text="WL Range:").pack(side=tk.LEFT)
        self.wf_wlim0 = tk.DoubleVar(value=0)  # 0 means auto/full
        self.wf_wlim1 = tk.DoubleVar(value=0)
        ttk.Entry(r3, textvariable=self.wf_wlim0, width=5).pack(side=tk.LEFT)
        ttk.Label(r3, text="-").pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.wf_wlim1, width=5).pack(side=tk.LEFT)

        ttk.Label(r3, text="| Norm:").pack(side=tk.LEFT, padx=(10, 0))
        self.wf_norm = tk.StringVar(value="None")
        # Added "Z-score" and "Min-Max" to the list
        ttk.OptionMenu(r3, self.wf_norm, "None", "None", "Max", "Area", "Z-score", "Min-Max").pack(side=tk.LEFT)
        ttk.Label(r3, text="| Cmap:").pack(side=tk.LEFT, padx=(10, 0))
        self.wf_cmap = tk.StringVar(value="viridis")
        # --- ADDED: turbo, inferno ---
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'jet', 'coolwarm']
        ttk.Combobox(r3, textvariable=self.wf_cmap, values=cmaps, width=8).pack(side=tk.LEFT)

        self.wf_offset = tk.DoubleVar(value=0.0)
        ttk.Label(r3, text="Off:").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Entry(r3, textvariable=self.wf_offset, width=4).pack(side=tk.LEFT)

        self.wf_show_fit = tk.BooleanVar(value=False)
        ttk.Checkbutton(r3, text="FitOverlay", variable=self.wf_show_fit).pack(side=tk.LEFT, padx=5)

        ttk.Button(r3, text="PLOT ►", style="Accent.TButton", command=self._wf_plot).pack(side=tk.LEFT, padx=15)

        # --- SPLIT SCREEN LAYOUT ---
        paned = tk.PanedWindow(tab, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === LEFT SIDE: Spatial Maps (Input & Output) ===
        self.wf_left_frame = ttk.Frame(paned)
        paned.add(self.wf_left_frame, width=400)

        # 1. Selector Map (Top-Left) - For clicking
        ttk.Label(self.wf_left_frame, text="1. Select Region (Click Start -> End)", font=("Segoe UI", 9, "bold")).pack(
            pady=(5, 0))

        from matplotlib.figure import Figure
        self.wf_map_fig = Figure(figsize=(4, 4), dpi=100)
        self.wf_map_canvas = FigureCanvasTkAgg(self.wf_map_fig, master=self.wf_left_frame)
        self.wf_map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.wf_map_canvas.mpl_connect('button_press_event', self._on_wf_selector_click)

        # Separator
        ttk.Separator(self.wf_left_frame, orient='horizontal').pack(fill='x', pady=5)

        # 2. Probe Path Map (Bottom-Left) - Shows Colored Dots
        ttk.Label(self.wf_left_frame, text="2. Probed Locations (Matches Waterfall Colors)",
                  font=("Segoe UI", 9, "bold")).pack(pady=(5, 0))

        self.wf_path_fig = Figure(figsize=(4, 4), dpi=100)
        self.wf_path_canvas = FigureCanvasTkAgg(self.wf_path_fig, master=self.wf_left_frame)
        self.wf_path_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === RIGHT SIDE: Spectral Results ===
        self.wf_right_frame = ttk.Frame(paned)
        paned.add(self.wf_right_frame)

        self.wf_res_fig = Figure(figsize=(8, 8), dpi=100)  # Taller figure
        self.wf_res_canvas = FigureCanvasTkAgg(self.wf_res_fig, master=self.wf_right_frame)
        self.wf_res_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.wf_res_canvas, self.wf_right_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._wf_click_state = 0
        self._draw_wf_selector_map()


    def _draw_wf_selector_map(self):
        """Draws the SEM image on the left and overlays the current selection vector."""
        try:
            # Get Data
            exp = self.wf_exp_var.get()
            hsp = self.cond_ans.data_dict[exp]
            data = hsp.get_numpy_spectra()
            n_rows, n_cols = data.shape[:2]
            img = getattr(hsp, 'get_live_scan', lambda: None)()

            # Setup Axes
            ax = self.wf_map_fig.gca()
            ax.clear()

            if img is None:
                ax.text(0.5, 0.5, "No Map Image", ha='center')
            else:
                ih, iw = img.shape[:2]
                ax.imshow(img, cmap='gray', extent=[0, iw, ih, 0])
                ax.set_title("Click 2 points: Start -> End")
                ax.axis('off')

                # Draw Selection Overlay
                r0, c0 = self.wf_r0.get(), self.wf_c0.get()
                r1, c1 = self.wf_r1.get(), self.wf_c1.get()

                cw, ch = iw / n_cols, ih / n_rows

                # Coordinates in image space
                x0, y0 = c0 * cw + cw / 2, r0 * ch + ch / 2
                x1, y1 = c1 * cw + cw / 2, r1 * ch + ch / 2

                if self.wf_mode.get() == "Line":
                    # Draw Arrow
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))
                    # Draw Start Dot
                    ax.plot(x0, y0, 'bo', markersize=6, label='Start')
                    # Draw End Dot
                    ax.plot(x1, y1, 'ro', markersize=6, label='End')
                else:
                    # Draw Rectangle
                    import matplotlib.patches as patches
                    # Determine top-left and width/height
                    xmin, xmax = sorted([x0, x1])
                    ymin, ymax = sorted([y0, y1])
                    w = xmax - xmin
                    h = ymax - ymin
                    # If points are identical, give it 1 pixel width visual
                    if w < 1: w = cw
                    if h < 1: h = ch

                    rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
                    ax.add_patch(rect)
                    ax.plot(x0, y0, 'bo', markersize=5)  # Show click points corner
                    ax.plot(x1, y1, 'ro', markersize=5)

            self.wf_map_canvas.draw()
        except Exception:
            pass

    def _on_wf_selector_click(self, event):
        """Handles clicks on the left map to set R0/C0 and R1/C1."""
        if event.inaxes != self.wf_map_fig.gca(): return

        try:
            # Convert click coord to grid coord
            exp = self.wf_exp_var.get()
            hsp = self.cond_ans.data_dict[exp]
            data = hsp.get_numpy_spectra()
            n_rows, n_cols = data.shape[:2]
            img = hsp.get_live_scan()

            ih, iw = img.shape[:2]
            cw, ch = iw / n_cols, ih / n_rows

            c = int(event.xdata / cw)
            r = int(event.ydata / ch)

            # Clip bounds
            c = max(0, min(c, n_cols - 1))
            r = max(0, min(r, n_rows - 1))

            if self._wf_click_state == 0:
                # State 0: User sets START point
                self.wf_r0.set(r)
                self.wf_c0.set(c)
                self._wf_click_state = 1  # Next click will be end
            else:
                # State 1: User sets END point
                self.wf_r1.set(r)
                self.wf_c1.set(c)
                self._wf_click_state = 0  # Reset to start

                # Optional: Trigger Plot automatically after 2nd click?
                # self._wf_plot()

            # Redraw the overlay
            self._draw_wf_selector_map()

        except Exception:
            pass

    def _wf_plot(self):
        """Draws plots. Updates Left-Bottom (Map) and Right (Waterfall/Heatmap)."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import lmfit

        # We now target TWO figures
        fig_res = self.wf_res_fig  # Right side
        fig_path = self.wf_path_fig  # Left-Bottom side

        try:
            # --- 1. Data Prep ---
            exp = self.wf_exp_var.get()
            hsp = self.cond_ans.data_dict[exp]
            data = hsp.get_numpy_spectra()
            wl = hsp.get_wavelengths()
            img = getattr(hsp, 'get_live_scan', lambda: None)()  # Get SE Image

            r0, c0 = self.wf_r0.get(), self.wf_c0.get()
            r1, c1 = self.wf_r1.get(), self.wf_c1.get()
            stride = self.wf_stride.get()
            mode = self.wf_mode.get()

            if mode == "Line":
                coords = self._bresenham_line(r0, c0, r1, c1)
                if stride > 1: coords = coords[::stride]
            else:
                coords = self._rect_coords(r0, c0, r1, c1, stride, "row-major")

            specs_raw, valid = self._extract_spectra(coords, data)
            if len(valid) == 0: return

            specs_raw = specs_raw.astype(float)

            # Wavelength Slicing
            w_min = self.wf_wlim0.get()
            w_max = self.wf_wlim1.get()
            if w_max > w_min:
                mask = (wl >= w_min) & (wl <= w_max)
                if np.sum(mask) > 1:
                    wl = wl[mask]
                    specs_raw = specs_raw[:, mask]

            # Normalize
            nm = self.wf_norm.get()
            specs = self._normalize_specs(specs_raw, nm)

            # Offset
            off = self.wf_offset.get()
            if off == 0.0:
                mx = np.max(specs, axis=1)
                off = 0.5 * np.median(mx) if len(mx) > 0 else 1.0

            # Fit Overlay Logic
            fits_overlay = []
            if self.wf_show_fit.get():
                peaks = self._collect_peak_params()
                if peaks:
                    composite = None
                    params_base = lmfit.Parameters()
                    if self.fit_bkg_flag:
                        composite = ConstantModel(prefix='bkg_')
                        params_base.update(composite.make_params(bkg_c=0))
                    for i, p in enumerate(peaks):
                        m = p['func'](prefix=f'p{i}_')
                        composite = m if composite is None else composite + m
                        mp = m.make_params()
                        for k in ['amplitude', 'center', 'sigma', 'gamma']:
                            if k in p: mp[f'p{i}_{k}'].set(value=p[k])
                        params_base.update(mp)

                    for y_raw in specs_raw:
                        try:
                            res = composite.fit(y_raw, params_base, x=wl)
                            y_fit = res.best_fit
                            if nm == "Max":
                                y_fit /= np.max(y_raw)
                            elif nm == "Area":
                                y_fit /= np.sum(y_raw)
                            fits_overlay.append(y_fit)
                        except:
                            fits_overlay.append(None)

            # --- SETUP COLORS ---
            N = len(valid)
            try:
                cmap = plt.get_cmap(self.wf_cmap.get())
            except:
                cmap = cm.get_cmap("viridis")
            colors = cmap(np.linspace(0, 1, N))

            # ==========================================
            # PART A: Draw "Probe Path" on Left-Bottom
            # ==========================================
            fig_path.clear()
            ax_path = fig_path.gca()

            if img is not None:
                ih, iw = img.shape[:2]
                # Extent fixes the alignment!
                ax_path.imshow(img, cmap='gray', extent=[0, iw, ih, 0])
                n_rows, n_cols = data.shape[:2]
                cw, ch = iw / n_cols, ih / n_rows

                # Calculate centers
                xs = [c * cw + cw / 2 for (_, c) in valid]
                ys = [r * ch + ch / 2 for (r, _) in valid]

                # Draw colored dots corresponding to spectra
                ax_path.scatter(xs, ys, c=np.arange(N), cmap=cmap, s=20, edgecolors='none', alpha=0.9)
                ax_path.set_title(f"Probed Pixels (N={N})")
                ax_path.axis('off')
            else:
                ax_path.text(0.5, 0.5, "No SE Image Available", ha='center')

            self.wf_path_canvas.draw()

            # ==========================================
            # PART B: Draw Results on Right Panel
            # ==========================================
            fig_res.clear()

            # Layout: Waterfall (Left), Heatmap/Max (Right)
            gs = fig_res.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
            ax_wf = fig_res.add_subplot(gs[:, 0])  # Waterfall
            ax_hm = fig_res.add_subplot(gs[0, 1])  # Heatmap
            ax_mx = fig_res.add_subplot(gs[1, 1])  # MaxInt

            # 1. Waterfall
            for i, y in enumerate(specs):
                ax_wf.plot(wl, y + i * off, color=colors[i], lw=1)
                if self.wf_show_fit.get() and i < len(fits_overlay) and fits_overlay[i] is not None:
                    ax_wf.plot(wl, fits_overlay[i] + i * off, 'k--', lw=0.8, alpha=0.7)
            ax_wf.set_title(f"Waterfall (N={N})")
            ax_wf.set_xlabel("Wavelength (nm)")
            ax_wf.set_yticks([])

            # 2. Heatmap (Inverted X, Y=Pixels)
            ax_hm.imshow(specs, aspect='auto', cmap=cmap, extent=[wl[0], wl[-1], N, 0])
            ax_hm.set_title("Heatmap")
            ax_hm.set_xlabel("Wavelength (nm)")
            ax_hm.set_ylabel("Pixel Index")
            ax_hm.invert_xaxis()  # Invert X as requested

            # 3. Max Intensity (Inverted axes: X=Pixels, Y=Int)
            mx_vals = np.max(specs, axis=1)
            ax_mx.plot(np.arange(N), mx_vals, 'k-', lw=0.8, alpha=0.5)
            ax_mx.scatter(np.arange(N), mx_vals, c=np.arange(N), cmap=cmap, s=15)
            ax_mx.set_title("Max Intensity")
            ax_mx.set_xlabel("Pixel Index")
            ax_mx.set_ylabel("Intensity")
            ax_mx.set_xlim(0, N)

            fig_res.tight_layout()
            self.wf_res_canvas.draw()

            self._wf_last_data = (specs_raw, valid, wl)

        except Exception as e:
            print(f"Waterfall Error: {e}")

    # ----------------------------------------------------------------------
    #                          EXPORT HELPER
    # ----------------------------------------------------------------------
    def _wf_export(self):
        if not hasattr(self, '_wf_last_data'):
            return
        try:
            specs, valid, wl = self._wf_last_data
            import pandas as pd

            fn = filedialog.asksaveasfilename(title="Save CSV", defaultextension=".csv")
            if not fn: return

            # Save Spectra with coordinates
            cols = [f"{w:.3f}" for w in wl]
            df = pd.DataFrame(specs, columns=cols)
            df.insert(0, 'row', [r for r, c in valid])
            df.insert(1, 'col', [c for r, c in valid])

            df.to_csv(fn, index=False)
            messagebox.showinfo("Export", f"Saved spectra to {fn}")
        except Exception as e:
            messagebox.showerror("Export", str(e))

    @staticmethod
    def _bresenham_line(r0, c0, r1, c1):
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        s_r = 1 if r0 < r1 else -1
        s_c = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        pts = []
        while True:
            pts.append((r, c))
            if r == r1 and c == c1: break
            e2 = 2 * err
            if e2 > -dc: err -= dc; r += s_r
            if e2 < dr: err += dr; c += s_c
        return pts

    @staticmethod
    def _rect_coords(r0, c0, r1, c1, stride=1, order="row-major"):
        rmin, rmax = sorted((int(r0), int(r1)))
        cmin, cmax = sorted((int(c0), int(c1)))
        rs = list(range(rmin, rmax + 1, max(1, int(stride))))
        cs = list(range(cmin, cmax + 1, max(1, int(stride))))
        coords = []
        if order == "row-major":
            for r in rs:
                for c in cs:
                    coords.append((r, c))
        else:
            for c in cs:
                for r in rs:
                    coords.append((r, c))
        return coords

    def _extract_spectra(self, coords, data):
        specs, valid = [], []
        n_r, n_c = data.shape[:2]
        for (r, c) in coords:
            if 0 <= r < n_r and 0 <= c < n_c:
                specs.append(data[r, c])
                valid.append((r, c))
        return np.asarray(specs), valid

    def _normalize_specs(self, specs, mode):
        """
        Normalizes spectra based on the selected mode.
        specs: 2D numpy array (N_pixels, N_wavelengths)
        """
        # Ensure we are working with floats
        specs = specs.astype(float)

        # 1. No Normalization
        if mode == "None":
            return specs

        # 2. Max Normalization (0 to 1, baseline preserved)
        elif mode == "Max":
            # Divide by the maximum intensity of each spectrum
            denom = np.max(specs, axis=1, keepdims=True)
            denom[denom == 0] = 1.0  # Avoid division by zero
            return specs / denom

        # 3. Area Normalization (Sum = 1)
        elif mode == "Area":
            # Divide by the total area (sum) of each spectrum
            denom = np.sum(specs, axis=1, keepdims=True)
            denom[denom == 0] = 1.0
            return specs / denom


        elif mode == "Z-score":
            mu = np.mean(specs, axis=1, keepdims=True)
            sigma = np.std(specs, axis=1, keepdims=True)
            sigma[sigma == 0] = 1.0
            return (specs - mu) / sigma

        # 5. Min-Max Normalization (Strictly 0 to 1)
        # (x - min) / (max - min)
        elif mode == "Min-Max":
            min_v = np.min(specs, axis=1, keepdims=True)
            max_v = np.max(specs, axis=1, keepdims=True)
            rng = max_v - min_v
            rng[rng == 0] = 1.0
            return (specs - min_v) / rng

        return specs

    def _export_peak_params(self):
        """Saves current peak definitions to a JSON file."""
        import json
        try:
            if not self.ss_peak_rows:
                messagebox.showinfo("Export", "No peaks to export.")
                return

            data = []
            for r in self.ss_peak_rows:
                row_data = {
                    "func": r['func'].get(),
                    "exists": r['exists'].get(),
                    "amp": r['amp'].get(), "amp_min": r['amp_min'].get(), "amp_max": r['amp_max'].get(),
                    "cen": r['cen'].get(), "cen_min": r['cen_min'].get(), "cen_max": r['cen_max'].get(),
                    "sig": r['sig'].get(), "sig_min": r['sig_min'].get(), "sig_max": r['sig_max'].get(),
                    "gam": r['gam'].get(), "gam_min": r['gam_min'].get(), "gam_max": r['gam_max'].get(),
                }
                data.append(row_data)

            fn = filedialog.asksaveasfilename(
                title="Save Parameters",
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All", "*.*")]
            )
            if fn:
                with open(fn, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Export", f"Saved {len(data)} peaks to {fn}")

        except Exception as e:
            messagebox.showerror("Export Failed", str(e))

    def _import_peak_params(self):
        """Loads peak definitions from a JSON file."""
        import json
        try:
            fn = filedialog.askopenfilename(
                title="Load Parameters",
                filetypes=[("JSON", "*.json"), ("All", "*.*")]
            )
            if not fn: return

            with open(fn, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid file format: root must be a list.")

            # 1. Clear existing rows
            while self.ss_peak_rows:
                self._remove_last_peak_row()

            # 2. Rebuild rows
            for entry in data:
                self._add_peak_row()
                r = self.ss_peak_rows[-1]  # Get the row we just created

                # Set Variables
                r['func'].set(entry.get('func', 'GaussianModel'))
                r['exists'].set(entry.get('exists', False))

                r['amp'].set(entry.get('amp', 1.0))
                r['amp_min'].set(entry.get('amp_min', 0.0))
                r['amp_max'].set(entry.get('amp_max', 0.0))

                r['cen'].set(entry.get('cen', 0.0))
                r['cen_min'].set(entry.get('cen_min', 0.0))
                r['cen_max'].set(entry.get('cen_max', 0.0))

                r['sig'].set(entry.get('sig', 1.0))
                r['sig_min'].set(entry.get('sig_min', 0.0))
                r['sig_max'].set(entry.get('sig_max', 0.0))

                r['gam'].set(entry.get('gam', 1.0))
                r['gam_min'].set(entry.get('gam_min', 0.0))
                r['gam_max'].set(entry.get('gam_max', 0.0))

                # 3. Trigger the event to enable/disable Gamma fields correctly
                if 'func_widget' in r:
                    r['func_widget'].event_generate("<<ComboboxSelected>>")

            self._ss_preview()  # Refresh plot
            messagebox.showinfo("Import", f"Loaded {len(data)} peaks.")

        except Exception as e:
            messagebox.showerror("Import Failed", f"Could not load parameters:\n{e}")

    def _build_tab_parameter_maps(self):
        """Builds the tab for visualizing fitting results (Excel/CSV) as maps."""
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Parameter Maps")

        # --- Controls Area ---
        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: Dataset (SE Image) & File Loading
        r1 = ttk.Frame(controls)
        r1.pack(fill=tk.X, pady=2)

        ttk.Label(r1, text="Reference Data (SE Image):").pack(side=tk.LEFT)
        self.pmap_exp_var = tk.StringVar(value=self.ref_key)
        keys = list(self.cond_ans.data_dict.keys())
        if self.ref_key not in keys and keys: self.pmap_exp_var.set(keys[0])
        ttk.OptionMenu(r1, self.pmap_exp_var, self.pmap_exp_var.get(), *keys,
                       command=lambda _: self._draw_param_map()).pack(side=tk.LEFT, padx=5)

        ttk.Button(r1, text="📂 Load Fit Results (Excel/CSV)...",
                   command=self._load_map_file).pack(side=tk.LEFT, padx=15)

        self.pmap_file_label = ttk.Label(r1, text="No file loaded", foreground="gray")
        self.pmap_file_label.pack(side=tk.LEFT)

        # Row 2: Parameter Selection & Styling
        r2 = ttk.Frame(controls)
        r2.pack(fill=tk.X, pady=2)

        ttk.Label(r2, text="Parameter:").pack(side=tk.LEFT)
        self.pmap_param_var = tk.StringVar()
        self.pmap_param_cb = ttk.Combobox(r2, textvariable=self.pmap_param_var, state="readonly", width=25)
        self.pmap_param_cb.pack(side=tk.LEFT, padx=5)
        self.pmap_param_cb.bind("<<ComboboxSelected>>", lambda _: self._draw_param_map())

        # --- NEW: eV Conversion Toggle ---
        self.pmap_convert_ev = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="nm → eV", variable=self.pmap_convert_ev,
                        command=lambda: self._draw_param_map()).pack(side=tk.LEFT, padx=5)

        ttk.Label(r2, text="| Colormap:").pack(side=tk.LEFT, padx=(10, 0))
        self.pmap_cmap = tk.StringVar(value="viridis")
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'jet', 'coolwarm', 'seismic']
        self.pmap_cmap_cb = ttk.Combobox(r2, textvariable=self.pmap_cmap, values=cmaps, width=10)
        self.pmap_cmap_cb.pack(side=tk.LEFT, padx=5)
        self.pmap_cmap_cb.bind("<<ComboboxSelected>>", lambda _: self._draw_param_map())

        ttk.Label(r2, text="| Alpha:").pack(side=tk.LEFT, padx=(10, 0))
        self.pmap_alpha = tk.DoubleVar(value=0.6)
        ttk.Scale(r2, variable=self.pmap_alpha, from_=0.0, to=1.0, command=lambda _: self._draw_param_map()).pack(
            side=tk.LEFT, padx=5)

        # Row 3: Range Limits (Clim)
        r3 = ttk.Frame(controls)
        r3.pack(fill=tk.X, pady=2)

        ttk.Label(r3, text="Color Range (Min - Max):").pack(side=tk.LEFT)
        self.pmap_vmin = tk.DoubleVar(value=0)
        self.pmap_vmax = tk.DoubleVar(value=0)

        self.pmap_en_min = ttk.Entry(r3, textvariable=self.pmap_vmin, width=8)
        self.pmap_en_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(r3, text="-").pack(side=tk.LEFT)
        self.pmap_en_max = ttk.Entry(r3, textvariable=self.pmap_vmax, width=8)
        self.pmap_en_max.pack(side=tk.LEFT, padx=2)

        self.pmap_auto_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(r3, text="Auto Scale", variable=self.pmap_auto_scale,
                        command=self._draw_param_map).pack(side=tk.LEFT, padx=10)

        ttk.Button(r3, text="Update Plot", command=self._draw_param_map).pack(side=tk.LEFT, padx=5)

        # --- Plot Area ---
        self.pmap_frame = ttk.Frame(tab)
        self.pmap_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        from matplotlib.figure import Figure
        self.pmap_fig = Figure(figsize=(8, 6), dpi=100)
        self.pmap_canvas = FigureCanvasTkAgg(self.pmap_fig, master=self.pmap_frame)
        self.pmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.pmap_canvas, self.pmap_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.pmap_df = None

    def _load_map_file(self):
        """Loads the Fit Results Excel/CSV file."""
        import pandas as pd
        import os

        fn = filedialog.askopenfilename(
            title="Open Fit Results",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All", "*.*")]
        )
        if not fn: return

        try:
            if fn.lower().endswith('.csv'):
                df = pd.read_csv(fn)
            else:
                df = pd.read_excel(fn)

            # Validation: Needs row, col columns
            if 'row' not in df.columns or 'col' not in df.columns:
                messagebox.showerror("Load Error", "File must contain 'row' and 'col' columns.")
                return

            self.pmap_df = df
            self.pmap_file_label.config(text=os.path.basename(fn))

            # Populate Parameter Dropdown (exclude non-numeric or structural cols)
            ignore = {'row', 'col', 'func_list', 'model', 'func'}
            params = [c for c in df.columns if c not in ignore]
            # Try to filter only numeric columns
            params = [c for c in params if pd.api.types.is_numeric_dtype(df[c])]

            self.pmap_param_cb['values'] = params
            if params:
                # Default to something interesting like center or amplitude
                defaults = [p for p in params if 'center' in p]
                if defaults:
                    self.pmap_param_cb.set(defaults[0])
                else:
                    self.pmap_param_cb.set(params[0])

            self._draw_param_map()

        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read file:\n{e}")

    def _draw_param_map(self):
        """Reconstructs the 2D map from the dataframe and plots it over the SE image."""
        if self.pmap_df is None: return

        import matplotlib.pyplot as plt
        import numpy as np

        param = self.pmap_param_var.get()
        if not param: return

        try:
            # 1. Get SE Image (Background)
            exp = self.pmap_exp_var.get()
            hsp = self.cond_ans.data_dict.get(exp)
            img = None
            if hsp:
                img = getattr(hsp, 'get_live_scan', lambda: None)()

            # 2. Reconstruct Parameter Grid
            max_r = self.pmap_df['row'].max()
            max_c = self.pmap_df['col'].max()

            # Grid of NaNs
            grid = np.full((max_r + 1, max_c + 1), np.nan)

            # Extract Data
            valid_df = self.pmap_df.dropna(subset=['row', 'col', param])
            rows = valid_df['row'].astype(int)
            cols = valid_df['col'].astype(int)
            vals = valid_df[param].values.astype(float)

            # --- NEW: eV Conversion Logic ---
            # Apply only if checkbox is checked AND parameter name implies it's a wavelength/center
            unit_label = param
            if self.pmap_convert_ev.get() and "center" in param.lower():
                # 1239.84193 / nm = eV
                # Handle 0 or NaN to avoid crash
                vals = np.divide(1239.84193, vals, out=np.full_like(vals, np.nan), where=vals != 0)
                unit_label = f"{param} (eV)"

            # Clip indices just in case
            rows = np.clip(rows, 0, max_r)
            cols = np.clip(cols, 0, max_c)
            grid[rows, cols] = vals

            # 3. Determine Color Limits
            if self.pmap_auto_scale.get():
                vmin = np.nanmin(grid)
                vmax = np.nanmax(grid)
                self.pmap_vmin.set(vmin)
                self.pmap_vmax.set(vmax)
            else:
                vmin = self.pmap_vmin.get()
                vmax = self.pmap_vmax.get()

            # 4. Plotting
            self.pmap_fig.clear()
            ax = self.pmap_fig.gca()

            # Draw SE Image (Background)
            if img is not None:
                ih, iw = img.shape[:2]
                ax.imshow(img, cmap='gray', extent=[0, iw, ih, 0])
                # Overlay Map
                ax.imshow(grid, cmap=self.pmap_cmap.get(), alpha=self.pmap_alpha.get(),
                          vmin=vmin, vmax=vmax, extent=[0, iw, ih, 0], interpolation='nearest')
            else:
                ax.imshow(grid, cmap=self.pmap_cmap.get(),
                          vmin=vmin, vmax=vmax, interpolation='nearest')

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=self.pmap_cmap.get(), norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = self.pmap_fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(unit_label)

            ax.set_title(f"Map: {unit_label}")
            ax.set_axis_off()

            self.pmap_canvas.draw()

        except Exception as e:
            print(f"Map Draw Error: {e}")