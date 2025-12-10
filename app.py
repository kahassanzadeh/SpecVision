import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from DataLoadingPanel import DataLoadingPanel
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

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Accent.TButton", foreground="white", background="#0078d4")

    app = DataLoadingPanel(root)
    root.mainloop()