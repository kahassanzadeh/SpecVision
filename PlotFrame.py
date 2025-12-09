import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from HspyPrep import HspyPrep
    from new_cd import CondAns
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


class PlotFrame(ttk.Frame):
    def __init__(self, master, figsize=(6, 3)):
        super().__init__(master)
        self.fig = plt.Figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()
