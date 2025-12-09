import tkinter as tk
from tkinter import ttk, messagebox
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



class Tooltip:
    """Simple tooltip that shows near the widget on hover/click."""

    def __init__(self, widget, text, *, delay=300, wraplength=250, follow_mouse=True):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self.follow_mouse = follow_mouse
        self._id = None
        self._tip = None
        self._x = self._y = 0

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._unschedule)
        widget.bind("<ButtonPress>", self._unschedule)
        if follow_mouse:
            widget.bind("<Motion>", self._follow)

    def _follow(self, event):
        self._x, self._y = event.x_root + 12, event.y_root + 12
        if self._tip is not None:
            self._tip.geometry(f"+{self._x}+{self._y}")

    def _schedule(self, _):
        self._unschedule(None)
        self._id = self.widget.after(self.delay, self._show)

    def _unschedule(self, _):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        self._hide()

    def _show(self):
        if self._tip is not None:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.attributes("-topmost", True)
        label = ttk.Label(
            self._tip,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padding=(8, 6),
            wraplength=self.wraplength,
        )
        label.pack()
        # Position
        if self._x == 0 and self._y == 0:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        else:
            x, y = self._x, self._y
        self._tip.geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

