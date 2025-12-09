import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import matplotlib.pyplot as plt
import traceback
from Tooltip import Tooltip
from PreprocessingPanel import PreprocessingPanel

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


class DataLoadingPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("CL/PL Analysis Loader - Step 1: Load Data")
        self.root.geometry("820x740")

        # State
        self.data_folder = tk.StringVar()
        self.bg_file = tk.StringVar()
        self.dict_of_files = {}
        self.load_mode = tk.StringVar(value="series")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        mode_frame = ttk.LabelFrame(main_frame, text="1. Select Loading Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Radiobutton(mode_frame, text="Comparative Series (CL)", variable=self.load_mode, value="series",
                        command=self.on_mode_change).pack(anchor="w", padx=10)
        ttk.Radiobutton(mode_frame, text="Single Experiment (CL/PL Map)", variable=self.load_mode, value="single",
                        command=self.on_mode_change).pack(anchor="w", padx=10)
        ttk.Radiobutton(mode_frame, text="Single Spectra (PL/CL) - [Not Implemented]", variable=self.load_mode,
                        value="spectra", state="disabled", command=self.on_mode_change).pack(anchor="w", padx=10)

        tip_bar = ttk.Frame(main_frame);
        tip_bar.pack(fill=tk.X, padx=5, pady=(4, 8))
        info_icon = ttk.Label(tip_bar, text="ⓘ", font=("Segoe UI", 11));
        info_icon.pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(tip_bar, text="Use Ctrl/Shift to multi‑select. Drag rows to reorder processing/analysis order.").pack(
            side=tk.LEFT)
        Tooltip(info_icon, text=(
            "Tips:\n• Ctrl/Shift to multi‑select.\n• Drag & drop rows to reorder.\n• In ‘Single’ mode choose an external background file."),
                wraplength=320)

        self.controls_frame = ttk.LabelFrame(main_frame, text="2. Select Data")
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        row1 = ttk.Frame(self.controls_frame);
        row1.pack(fill=tk.X, padx=5, pady=5)
        self.btn_folder = ttk.Button(row1, text="Select Root Folder", command=self.select_data_folder);
        self.btn_folder.pack(side=tk.LEFT, padx=5)
        self.lbl_folder = ttk.Label(row1, textvariable=self.data_folder, relief="sunken", width=70);
        self.lbl_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        row2 = ttk.Frame(self.controls_frame);
        row2.pack(fill=tk.X, padx=5, pady=5)
        self.btn_bg = ttk.Button(row2, text="Select Background File", command=self.select_bg_file);
        self.btn_bg.pack(side=tk.LEFT, padx=5)
        self.lbl_bg = ttk.Label(row2, textvariable=self.bg_file, relief="sunken", width=70);
        self.lbl_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        row3 = ttk.Frame(self.controls_frame);
        row3.pack(fill=tk.X, padx=5, pady=5)
        self.btn_load = ttk.Button(row3, text="Load Data", command=self.load_data, style="Accent.TButton");
        self.btn_load.pack(side=tk.RIGHT, padx=10)

        self.list_frame = ttk.LabelFrame(main_frame, text="3. Detected Experiments")
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        list_inner = ttk.Frame(self.list_frame);
        list_inner.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(list_inner, columns=("Status"), height=12, selectmode="extended")
        self.tree.heading("#0", text="Experiment Name");
        self.tree.heading("Status", text="Status")
        self.tree.column("#0", width=340);
        self.tree.column("Status", width=240)
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)
        list_button_frame = ttk.Frame(list_inner);
        list_button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(list_button_frame, text="Verify Selected", command=self.verify_selected).pack(side=tk.TOP, padx=5,
                                                                                                 pady=(5, 2))
        ttk.Button(list_button_frame, text="Rename Selected", command=self.rename_selected).pack(side=tk.TOP, padx=5,
                                                                                                 pady=2)

        info_btn = ttk.Label(self.list_frame, text="ⓘ", font=("Segoe UI", 11))
        info_btn.place(relx=1.0, x=-24, y=4)
        Tooltip(info_btn,
                text=("Drag a row and drop it above/below another to reorder.\nThe visual order is used later."))

        self._dragging_iid = None
        self._setup_tree_drag_and_drop()

        action_frame = ttk.LabelFrame(main_frame, text="4. Preprocessing")
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_status = ttk.Label(action_frame, text="Status: Waiting for data...");
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        self.btn_preprocess = ttk.Button(action_frame, text="Proceed to Preprocessing →",
                                         command=self.open_preprocessing_panel, state="disabled")
        self.btn_preprocess.pack(side=tk.RIGHT, padx=10, pady=5)

        self.on_mode_change()

    # ---- DnD for Treeview ----
    def _setup_tree_drag_and_drop(self):
        self.tree.bind('<ButtonPress-1>', self._on_tree_button_press)
        self.tree.bind('<B1-Motion>', self._on_tree_mouse_drag)
        self.tree.bind('<ButtonRelease-1>', self._on_tree_button_release)

    def _on_tree_button_press(self, event):
        rowid = self.tree.identify_row(event.y)
        self._dragging_iid = rowid if rowid else None

    def _on_tree_mouse_drag(self, event):
        if not self._dragging_iid:
            return
        self.tree.after(1, self._drag_feedback, event)

    def _drag_feedback(self, event):
        y = event.y
        target = self.tree.identify_row(y)
        if not target or target == self._dragging_iid:
            return
        bbox = self.tree.bbox(target)
        where = 'above'
        if bbox:
            mid_y = bbox[1] + bbox[3] // 2
            where = 'above' if y < mid_y else 'below'
        try:
            self.tree.move(self._dragging_iid, '', self._index_for_move(target, where))
        except tk.TclError:
            pass

    def _index_for_move(self, target, where):
        children = list(self.tree.get_children(''))
        t_index = children.index(target)
        return t_index if where == 'above' else t_index + 1

    def _on_tree_button_release(self, _):
        self._dragging_iid = None

    def on_mode_change(self):
        mode = self.load_mode.get()
        if mode == "series":
            self.btn_folder.config(text="Select Root Folder")
            self.btn_bg.config(state="disabled")
            self.lbl_bg.config(relief="flat")
            self.bg_file.set("")
            self.list_frame.config(text="3. Detected Experiments")
        elif mode == "single":
            self.btn_folder.config(text="Select Experiment Folder")
            self.btn_bg.config(state="normal")
            self.lbl_bg.config(relief="sunken")
            self.list_frame.config(text="3. Detected Experiment")
        elif mode == "spectra":
            self.btn_folder.config(text="Select Spectra File")
            self.btn_bg.config(state="normal")
            self.lbl_bg.config(relief="sunken")
            self.list_frame.config(text="3. Detected Spectrum")

    def select_data_folder(self):
        mode = self.load_mode.get()
        if mode == "series":
            path = filedialog.askdirectory(title="Select the Root Folder")
        elif mode == "single":
            path = filedialog.askdirectory(title="Select a Single Experiment Folder")
        elif mode == "spectra":
            path = filedialog.askopenfilename(title="Select Spectra File")
        else:
            path = None
        if path: self.data_folder.set(path)

    def select_bg_file(self):
        path = filedialog.askopenfilename(title="Select Background File (e.g., BG.txt)",
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path: self.bg_file.set(path)

    def load_data(self):
        if not self.data_folder.get():
            messagebox.showerror("Error", "Please select a data folder/file first.");
            return
        self.lbl_status.config(text="Loading... Please wait.")
        self.btn_load.config(state="disabled")
        self.tree.delete(*self.tree.get_children());
        self.dict_of_files.clear()
        try:
            mode = self.load_mode.get()
            if mode == "series":
                self.load_series_experiment()
            elif mode == "single":
                self.load_single_experiment()
            elif mode == "spectra":
                messagebox.showinfo("Not Implemented", "Loading single spectra files is not yet supported.")
            if self.dict_of_files: self.btn_preprocess.config(state="normal")
        except Exception as e:
            messagebox.showerror("Loading Failed", f"An error occurred: {e}\n{traceback.format_exc()}");
            self.lbl_status.config(text="Status: Error.")
        self.btn_load.config(state="normal")

    def load_series_experiment(self):
        root_path = self.data_folder.get()
        for file_name in os.listdir(root_path):
            if 'HYP' in file_name:
                exp_path = os.path.join(root_path, file_name, '')
                try:
                    hsp_obj = HspyPrep(exp_path, step=1, whole_seconds=64 * 64, contain_bg=True)
                    self.dict_of_files[file_name] = hsp_obj
                    self.tree.insert("", "end", text=file_name, values=("Loaded (Raw)",), iid=file_name)
                except Exception as e:
                    self.tree.insert("", "end", text=file_name, values=(f"Error: {e}",))
        self.lbl_status.config(text=f"Series loading complete. {len(self.dict_of_files)} experiments found.")

    def load_single_experiment(self):
        exp_path = self.data_folder.get()
        exp_name = os.path.basename(os.path.normpath(exp_path))
        try:
            hsp_obj = HspyPrep(exp_path + '/', step=1, whole_seconds=64 * 64, contain_bg=False)
            self.dict_of_files[exp_name] = hsp_obj
            self.tree.insert("", "end", text=exp_name, values=("Loaded (Raw)",), iid=exp_name)
        except Exception as e:
            self.tree.insert("", "end", text=exp_name, values=(f"Error: {e}",))
        self.lbl_status.config(text=f"Single experiment '{exp_name}' loaded.")

    def _ordered_selection(self):
        selected = set(self.tree.selection())
        return [iid for iid in self.tree.get_children("") if iid in selected]

    def verify_selected(self):
        selected_items = self.tree.selection()
        if not selected_items: messagebox.showinfo("Info", "Please select an experiment to verify."); return
        first_item_id = self._ordered_selection()[0]
        exp_name = self.tree.item(first_item_id, "text")
        hsp_obj = self.dict_of_files.get(exp_name)
        if not hsp_obj: messagebox.showerror("Error", f"Could not find loaded data for {exp_name}."); return
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.imshow(hsp_obj.get_live_scan(), cmap='gray')
            ax1.set_title(f"{exp_name} - Live Scan")
            ax1.grid(False)
            data = hsp_obj.get_numpy_spectra()
            cy, cx = data.shape[0] // 2, data.shape[1] // 2
            spectrum = data[cy, cx, :]
            wavelengths = hsp_obj.get_wavelengths()
            ax2.plot(wavelengths, spectrum)
            ax2.set_title(f"Sample Spectrum @ ({cy}, {cx})")
            ax2.set_xlabel("Wavelength (nm)")
            ax2.set_ylabel("Intensity (a.u.)")
            ax2.grid(True)
            fig.suptitle(f"Verification for {exp_name}")
            fig.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not create verification plot: {e}\n{traceback.format_exc()}")

    def rename_selected(self):
        selected_items = self.tree.selection()
        if not selected_items: messagebox.showinfo("Info", "Please select an experiment to rename."); return
        if len(selected_items) > 1: messagebox.showinfo("Info", "Please select only one experiment to rename."); return
        item_id = selected_items[0]
        old_name = self.tree.item(item_id, "text")
        new_name = simpledialog.askstring("Rename Experiment", f"Enter new name for '{old_name}':", parent=self.root)
        if new_name and new_name != old_name:
            if new_name in self.dict_of_files: messagebox.showerror("Error",
                                                                    f"The name '{new_name}' already exists."); return
            values = self.tree.item(item_id, 'values');
            index = list(self.tree.get_children('')).index(item_id)
            self.tree.delete(item_id)
            self.tree.insert('', index, text=new_name, values=values, iid=new_name)
            hsp_obj = self.dict_of_files.pop(old_name)
            self.dict_of_files[new_name] = hsp_obj

    def open_preprocessing_panel(self):
        selected_item_ids = self.tree.selection()
        if not selected_item_ids: messagebox.showerror("Error",
                                                       "Please select one or more experiments to process."); return
        ordered_ids = self._ordered_selection()
        analysis_dict = {}
        selected_names = []
        for iid in ordered_ids:
            exp_name = self.tree.item(iid, "text")
            if exp_name in self.dict_of_files:
                analysis_dict[exp_name] = self.dict_of_files[exp_name]
                selected_names.append(exp_name)
        if not analysis_dict: messagebox.showerror("Error", "Could not find data for selected items."); return
        try:
            ref_key = selected_names[0]
            self.root.withdraw()
            PreprocessingPanel(self.root, analysis_dict, ref_key, self.bg_file.get(), self.load_mode.get())
        except Exception as e:
            messagebox.showerror("Initialization Error",
                                 f"Failed to open preprocessing panel: {e}\n{traceback.format_exc()}")
