# 🚀 SpecVision: A Platform for Advanced Computational Spectroscopy

![SpecVision Logo](logo.png) 

# SpecVision - High-Precision Luminescence Spectroscopy Toolkit

## 1. Introduction

### 1.1 What is SpecVision?

SpecVision is a desktop application for analysing **cathodoluminescence (CL)** and **photoluminescence (PL)** hyperspectral datasets.  
It was designed in a research environment to:

- Load and manage multiple CL/PL experiments
- Preprocess spectra (background removal and smoothing)
- Explore spectra pixel-by-pixel
- Fit emission peaks using standard line-shape models
- Export quantitative results for further analysis (e.g. in Excel, Origin, Python, R)

No programming knowledge is required to use SpecVision.

### 1.2 Typical Use Cases

- Comparing CL maps of a sample at different accelerating voltages
- Studying temperature-dependent emission
- Extracting peak positions, widths (FWHM) and intensities across a field of view
- Building intensity / parameter maps for different emission bands

## 2. Installation

### 2.1 Requirements

- A computer running **Windows, macOS or Linux**
- **Python 3.11** installed
- Internet connection (only for the first installation to download packages)

### 2.2 Install Python 3.11

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download Python 3.11 for your operating system.
3. On Windows, during installation, tick the box:
   - **"Add Python to PATH"**

### 2.3 Download SpecVision (Git)

1. Open **Terminal** (macOS/Linux) or **Command Prompt** (Windows).
2. Run:
   ```bash
   git clone https://github.com/kahassanzadeh/SpecVision.git
   ```
3. A new folder named SpecVision will appear.

### 2.4 Install Required Packages

1. In the same Terminal / Command Prompt window, move into the folder:
   ```bash
   cd SpecVision
   ```
2. Install all the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

This step may take a few minutes the first time (Hyperspy and Lumispy are relatively large).

### 2.5 Launch the Application

Still inside the SpecVision folder, run:

```bash
python app.py
```

The main SpecVision window will open. From now on, you work entirely in the graphical interface.

## 3. Data Requirements

### 3.1 Experiment Folder Structure

SpecVision is designed for datasets produced by HYPCard systems.  
Each experiment should be stored in a folder with at least:

- HYPCard.sur - hyperspectral data cube
- Live_XXXXX.sur - live scan image (for reference and verification)

Optional files:

- SE_Before-SE.sur - SEM image before measurement
- SE_After-SE.sur - SEM image after measurement

Example:

```
YourFolder/
├── HYP-THICK-PEELED-5KEV/
│   ├── HYPCard.sur
│   ├── Live_00001.sur
│   ├── SE_Before-SE.sur (optional)
│   └── SE_After-SE.sur (optional)
│
└── HYP-THICK-PEELED-10KEV/
    ├── HYPCard.sur
    └── Live_00001.sur
```

**Important rule:**

> SpecVision recognises a folder as an experiment **only if that folder contains HYPCard.sur.**

Other files (JPG/PNG/SVG/DESC/STATUS) are ignored and do not disturb the analysis.

### 3.2 Background Files

You may provide an external background spectrum, for example:

- BG-xxx.txt

The background file can be placed:

- In the **root dataset folder** (global background), or
- Inside a particular experiment folder (experiment-specific background)

During preprocessing, you can tell SpecVision whether to subtract background and how to smooth the spectra.

## 4. Quick Start Workflow

1. Prepare your data as described in Section 3.
2. Start SpecVision (`python app.py`).
3. **Step 1 - Load Experiments:** point to the root folder that contains the experiment subfolders.
4. **Step 2 - Preprocessing:** preview a pixel, enable background removal and smoothing, then apply to all.
5. **Step 3 - Analysis:** explore spectra, fit peaks, select ROIs and export to Excel.

The next sections describe each step in more detail.

## 5. Step 1 - Loading Experiments

When SpecVision starts, you see **"CL/PL Analysis Loader - Step 1: Load Data"**.

Main elements on this screen:

- **Loading Mode**
  - *Comparative Series (CL)* - load multiple experiments (e.g. different voltages).
  - *Single Experiment (CL/PL Map)* - work with a single hyperspectral map.
- **Select Root Folder** - choose the folder that contains all experiment folders.
- **Load Data** button - scans the root folder and lists all detected experiments.
- **Detected Experiments table** - shows experiment names and their loading status.
- **Verify Selected** - opens a small preview (LiveScan + sample spectrum).
- **Rename Selected** - allows you to rename experiments.
- **Drag & Drop** in the table - change processing / analysis order.

### 5.1 Typical Procedure

1. Choose **Comparative Series (CL)** if you have more than one experiment.
2. Click **Select Root Folder** and select your dataset directory.
3. Press **Load Data**.
4. Check that all expected experiments appear with status **"Loaded (Raw)"**.
5. Optionally:
   - Use **Verify Selected** to visually confirm each experiment.
   - Use **Rename Selected** to assign more readable names.
   - Drag experiments up/down to set their order.
6. Click **Proceed to Preprocessing →**.

## 6. Step 2 - Preprocessing

The Preprocessing window shows:

- A dropdown to select the **experiment**.
- Inputs to select **Row** and **Column** of a pixel.
- A **Preview** plot with:
  - The raw spectrum.
  - The processed spectrum (after background removal and smoothing).
- A checkbox **Remove Background**.
- A slider **Smoothing Kernel Size (must be odd)**.
- Buttons **APPLY TO ALL** and **Proceed to Analysis →**.

### 6.1 What Preprocessing Does

- **Background Removal:**
  - Subtracts a background spectrum (either loaded from file or estimated).
  - Useful to remove system response or constant offset.
- **Smoothing:**
  - Uses a median filter with a kernel of odd size (e.g. 11, 13, 15).
  - Reduces noise while preserving peak positions and widths reasonably well.

### 6.2 Recommended Workflow

1. Select a representative experiment and pixel (Row, Col).
2. Click **Load Sample Spectrum** (if available) or **Preview**.
3. Check the *Raw* spectrum.
4. Tick **Remove Background**.
5. Move the smoothing slider until the previewed spectrum looks clean but not over-smoothed.
6. When satisfied, click **APPLY TO ALL**:
   - The same preprocessing configuration is applied to all loaded experiments.
7. Continue with **Proceed to Analysis →**.

## 7. Step 3 - Analysis & Peak Fitting

The Analysis panel includes:

- Tabs:
  - **Single Spectrum**
  - **Waterfall Plots**
  - **Parameter Maps**
- On the Single Spectrum tab:
  - Experiment selector
  - Row/Col selectors with arrow buttons
  - **Preview** button
  - Checkboxes: **Live Fit**, **Show Components**, **Automatic Background Fit**
  - Plot of the current spectrum
  - Map of the sample (for spatial navigation)
  - Peak-table / parameter grid
  - Buttons: **+ Add Peak**, **Remove Last**, **Import Params**, **Export Params**, **Fit Selection**, **Fit All**
  - ROI selection controls (Select ROI, row/col start/end)
  - Field for **R² threshold**

### 7.1 Navigating Spectra

You can move between spectra using:

- The **Row** and **Col** arrow buttons.
- The **arrow keys** on your keyboard.
- Clicking on the pixel in the **sample map** on the right.

After choosing a pixel, click **Preview** (or enable **Live Fit**) to update.

### 7.2 Manual Peak Fitting

At the bottom, there is a table of peaks. Each row corresponds to one peak.

You can:

- Choose model type from a dropdown: *GaussianModel*, *LorentzianModel*, *VoigtModel*.
- Set parameters:
  - amplitude (A)
  - center (μ)
  - sigma (σ)
  - optional min/max bounds

Buttons:

- **+ Add Peak** - append a new peak to the model.
- **-- Remove Last** - delete the last peak.
- **Fit** - performs the fit and updates R² and component plots.

When **Live Fit** is active, changes are applied automatically.

### 7.3 Region of Interest (ROI) & Batch Fitting

You can fit more than one pixel at once:

1. Use **Select ROI (draw)** to drag a rectangle on the sample map.
2. Use **Fit Selection** to fit only pixels inside the ROI.
3. Use **Fit All** to fit *every* pixel in the dataset.

The **R² threshold** field can be used to filter out bad fits (for example, only keep fits with R² ≥ 0.9).

### 7.4 Exporting Results

After fitting:

- Click the export button (if present in your version) or the corresponding menu action.
- Choose a filename and location for the Excel file (.xlsx).

The file will contain one row per pixel with columns such as:

- row, col
- peak1_center, peak1_fwhm, peak1_amp, peak1_area
- peak2_center, ... (for additional peaks)
- r2
- Background parameters (e.g. bkg_c0, bkg_c1, ...)

You can open this file in Excel, Origin, or any analysis tool.

## 8. Troubleshooting

### 8.1 The program does not start

- Make sure you installed Python 3.11.
- Check that packages are installed:
  ```bash
  pip show hyperspy
  ```
- If missing, reinstall:
  ```bash
  pip install -r requirements.txt
  ```

### 8.2 Hyperspy cannot read .sur files

- Try installing a specific version of Hyperspy that is known to work well:
  ```bash
  pip install "hyperspy<2.0"
  ```
- Confirm that .sur files are not corrupted and come directly from the acquisition system.

### 8.3 No experiments are detected

- Ensure each experiment folder contains **HYPCard.sur**.
- Ensure you selected the **root folder**, not an experiment subfolder.
- Folder names may contain spaces or hyphens; this is fine.

### 8.4 Background not applied

- Check that **Remove Background** is ticked in Step 2.
- Make sure the background file is named like BG-*.txt and placed at the correct level.
- If no external BG file is available, SpecVision may estimate a background internally (depending on configuration).

## 9. For Developers (Optional)

Although SpecVision is meant to be used without coding, developers may be interested in the internal structure.

Main components:

- `app.py` - launches the main Tkinter window, assembles panels.
- `DataLoadingPanel.py` - handles step 1: experiment discovery, verification, reordering.
- `PreprocessingPanel.py` - manages background subtraction and smoothing.
- `AnalysisPanel.py` - contains the main fitting UI and logic, ROI selection, map display.
- `HspyPrep.py` - backend utilities for loading .sur files, converting Hyperspy objects into NumPy arrays, performing background calculation and filtering.
- `PlotFrame.py` - reusable widget wrapping a Matplotlib canvas inside Tkinter.
- `Tooltip.py` - small helper for hover-tooltips.
- `SpecVision.py` - older notebook-style implementation used as reference.

## 10. Summary

- **SpecVision** = High-precision luminescence spectroscopy toolkit.
- Designed for **CL/PL hyperspectral** datasets.
- Three main steps:
  1. Load experiments
  2. Preprocess spectra
  3. Analyse & fit peaks
- No programming needed; everything is handled through a graphical interface.
- Results are exported to Excel and can be used in any further analysis pipeline.