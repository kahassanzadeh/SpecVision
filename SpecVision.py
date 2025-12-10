from IPython.lib.display import FileLink
from lmfit import Parameters

# from hspy_utils import HspyPrep
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntProgress
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import matplotlib.patches as patches
import pickle
from lmfit.models import GaussianModel, ConstantModel, LorentzianModel, VoigtModel, LognormalModel, ExponentialGaussianModel
from matplotlib.gridspec import GridSpec
import traceback
import seaborn as sns
from scipy.signal import find_peaks, peak_widths
import os
import pandas as pd
from skimage.exposure import match_histograms
from ipywidgets import FloatSlider, IntSlider, Dropdown, Button, HBox, VBox, Output, Layout, BoundedIntText, FloatText, \
    BoundedFloatText, Checkbox, interact
from IPython.display import display
import re
from ipywidgets import HBox, VBox, Button, Dropdown, FloatText, BoundedIntText, Output, Layout, GridBox, HTML
import numpy as np
import matplotlib.pyplot as plt


class CondAns:
    def __init__(self, data_dict: dict, ref, addr_file, load_mapping=False):
        self.best_r2 = None
        self.best_result = None
        self.best_model = None
        self.params_fit = None
        self.best_fit = None
        self.data_dict = data_dict
        self.ref = ref
        self.list_of_exp = list(self.data_dict.keys())

        if load_mapping:
            with open(f"{addr_file}/data_coordinates.pkl", "rb") as file:
                self.data_coordinates = pickle.load(file)

    def map_all_pixels(self, window_size, max_disp, ref):
        image_ref = self.data_dict[ref].get_live_scan()
        mapping_save = dict()
        mapping = ''
        for key, value in self.data_dict.items():
            value.live_scan = match_histograms(value.live_scan, image_ref)
        for key in self.data_dict.keys():
            if key == ref:
                mapping_save['ref'] = key
                continue
            mapping = CondAns.map_pixels(image_ref, self.data_dict[key].get_live_scan(), window_size=window_size,
                                         search_radius=max_disp)
            mapping_save[key] = mapping
        print(mapping_save)
        mapping_save['ref'] = list(mapping.keys())
        with open("data_coordinates.pkl", "wb") as file:
            pickle.dump(mapping_save, file)
            self.data_coordinates = mapping_save

    def plot_all_pixels(self, figsize=(20, 10), save=False, filename=None, x_pixel=0, y_pixel=0, show_plots=False):
        '''

        :param show_plots: 
        :param figsize:
        :param save:
        :param filename:
        :param x_pixel:
        :param y_pixel:
        '''

        shape_image = self.data_dict[self.ref].get_live_scan().shape
        for key_coord in self.data_coordinates['ref']:
            key_coord = (key_coord[0] + y_pixel, key_coord[1] + x_pixel)
            if key_coord[1] > shape_image[1] or key_coord[0] > shape_image[0]:
                continue
            print(key_coord[0], key_coord[1])

            ax_main, image_axes, colors = self.__setup_plotting(figsize,
                                                                self.data_dict[self.ref].get_wavelengths()[::-1])
            coord_x, coord_y = self.__setup_coordinations(key_coord, image_axes)

            for idx, (temp, color, x, y) in enumerate(zip(list(self.data_dict.keys()), colors, coord_x, coord_y)):
                if x == 0 and y == 0:
                    continue
                wavelengths = self.data_dict[temp].get_wavelengths()[::-1]
                intensity = self.data_dict[temp].get_numpy_spectra()[y][x][::-1]
                ax_main.plot(wavelengths, intensity, color=color, linewidth=2, label=f'{temp} nA')

            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}.png', dpi=300)

            if show_plots:
                plt.show()
                plt.close()

    def plot_all_pixels_with_fitting(self, figsize=(20, 10), save=False, filename=None, fit_func=VoigtModel,
                                     peaks='Automatic', max_peaks=3, x_pixel=0, y_pixel=0, height=100, prominence=1,
                                     distance=5, show_plots=False):
        '''

        :param show_plots:
        :param x_pixel:
        :param y_pixel:
        :param prominence:
        :param height:
        :param distance:
        :param max_peaks:
        :param figsize:
        :param save:
        :param filename:
        :param fit_func: define a fit function which the data can be fitted with thats
        :param peaks: The number of peaks will be defined automatically; however, you can define estimations about the number of peaks that you may have.
        :return:
        '''

        shape_image = self.data_dict[self.ref].get_live_scan().shape
        for key_coord in self.data_coordinates['ref']:
            key_coord = (key_coord[0] + y_pixel, key_coord[1] + x_pixel)
            if key_coord[1] > shape_image[1] or key_coord[0] > shape_image[0]:
                continue

            print(key_coord[0], key_coord[1])

            ax_main, image_axes, colors = self.__setup_plotting(figsize,
                                                                self.data_dict[self.ref].get_wavelengths()[::-1])
            coord_x, coord_y = self.__setup_coordinations(key_coord, image_axes)

            for idx, (temp, color, x, y) in enumerate(zip(list(self.data_dict.keys()), colors, coord_x, coord_y)):
                if x == 0 and y == 0:
                    continue
                wavelengths = self.data_dict[temp].get_wavelengths()[::-1]
                intensity = self.data_dict[temp].get_numpy_spectra()[y][x][::-1]
                ax_main.plot(wavelengths, intensity, color=color, linewidth=2, label=f'{temp} nA')
                if peaks == 'Automatic':
                    self.__peak_fitting_auto(intensity, wavelengths, height, prominence, distance, max_peaks, fit_func)
                    ax_main.plot(wavelengths, self.best_result.best_fit, '--', color=color, linewidth=2,
                                 label=f'R²={self.best_r2:.4f}')
                if peaks == 'Manual':
                    pass

            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=300)

            if show_plots:
                plt.show()
                plt.close()

    def single_exp_run_plot(self, exp_key, figsize=(20, 10), save=False, show_plots=False, filename=None, x_pixel=0,
                            y_pixel=0):

        shape_image = self.data_dict[self.ref].get_live_scan().shape

        for key_coord in self.data_coordinates[exp_key]:
            key_coord = (key_coord[0] + y_pixel, key_coord[1] + x_pixel)
            if key_coord[1] > shape_image[1] or key_coord[0] > shape_image[0]:
                continue
            print(key_coord[0], key_coord[1])

            x = key_coord[1]
            y = key_coord[0]

            wavelengths = self.data_dict[exp_key].get_wavelengths()[::-1]
            intensity = self.data_dict[exp_key].get_numpy_spectra()[y][x][::-1]

            ax_main, ax_image = self.__setup_plotting_single_image(figsize, wavelengths)
            ax_main.plot(wavelengths, intensity, color='red', linewidth=2)
            CondAns.__plot_image_with_rect(ax_image, self.data_dict[exp_key].get_live_scan(),
                                           (x, y), exp_key)

            if save:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=300)

            if show_plots:
                plt.show()
                plt.close()

    def single_exp_plot(self, exp_key, x_pixel, y_pixel, figsize=(20, 10), save=False, show_plots=False, filename=None):

        shape_image = self.data_dict[self.ref].get_live_scan().shape

        # for key_coord in self.data_coordinates[exp_key]:
        #     key_coord = (key_coord[0] + y_pixel, key_coord[1] + x_pixel)
        #     if key_coord[1] > shape_image[1] or key_coord[0] > shape_image[0]:
        #         continue
        #     print(key_coord[0], key_coord[1])

        x = x_pixel
        y = y_pixel

        wavelengths = self.data_dict[exp_key].get_wavelengths()[::-1]
        intensity = self.data_dict[exp_key].get_numpy_spectra()[y][x][::-1]

        ax_main, ax_image = self.__setup_plotting_single_image(figsize, wavelengths)
        ax_main.plot(wavelengths, intensity, color='red', linewidth=2)
        CondAns.__plot_image_with_rect(ax_image, self.data_dict[exp_key].get_live_scan(),
                                       (x, y), exp_key)

        # if save:
        #     folder = f'./{filename}'
        #     os.makedirs(folder, exist_ok=True)
        #     plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=300)

        # if show_plots:
        plt.show()
        # plt.close()

    def single_exp_run_fitting(self, exp_key, figsize=(20, 10), save_excel=False, filename=None, fit_func=VoigtModel,
                               peaks='Automatic', max_peaks=3, x_pixel=0, y_pixel=0, height=100, prominence=1,
                               distance=5, save_plots=False):

        shape_image = self.data_dict[self.ref].get_live_scan().shape
        params_data = []

        for key_coord in self.data_coordinates[exp_key]:
            key_coord = (key_coord[0] + y_pixel, key_coord[1] + x_pixel)
            if key_coord[1] > shape_image[1] or key_coord[0] > shape_image[0]:
                continue
            print(key_coord[0], key_coord[1])

            x = key_coord[1]
            y = key_coord[0]

            wavelengths = self.data_dict[exp_key].get_wavelengths()[::-1]
            intensity = self.data_dict[exp_key].get_numpy_spectra()[y][x][::-1]

            ax_main, ax_image = self.__setup_plotting_single_image(figsize, wavelengths)
            ax_main.plot(wavelengths, intensity, color='red', linewidth=2)
            CondAns.__plot_image_with_rect(ax_image, self.data_dict[exp_key].get_live_scan(),
                                           (x, y), exp_key)

            if peaks == 'Automatic':
                self.__peak_fitting_auto(intensity, wavelengths, height, prominence, distance, max_peaks, fit_func)
                ax_main.plot(wavelengths, self.best_result.best_fit, '--', color=color, linewidth=2,
                             label=f'R²={self.best_r2:.4f}')
                try:
                    params_data.append({
                        "Key": (x, y),
                        "R^2": self.best_r2
                    })
                    for param_name, param in self.best_result.params.items():
                        params_data[-1][param_name] = param.value
                except Exception as e:
                    print("An error occurred during fitting:")
                    traceback.print_exc()

            if save_plots:
                folder = f'./{filename}'
                os.makedirs(folder, exist_ok=True)
                plt.savefig(f'{folder}/{key_coord[0]}_{key_coord[1]}_fitted.png', dpi=300)
                plt.show()
                plt.close()

            if save_excel:
                params_df = pd.DataFrame(params_data)
                params_df.to_excel(f"lmfit_parameters_{exp_key}.xlsx", index=False)

    def get_data_coordinate(self):
        return self.data_coordinates

    def __setup_plotting(self, figsize, wavelengths):
        colors = sns.color_palette('inferno', len(self.list_of_exp))
        n_images = len(self.list_of_exp)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(6, 7, figure=fig, wspace=0.1, hspace=0.2)
        ax_main = fig.add_subplot(gs[:, 0:4])
        image_axes = dict()
        n_cols = 3

        for i, (key, value) in enumerate(self.data_dict.items(), start=0):
            row = (i // n_cols) * 2
            col = 4 + (i % n_cols)
            ax_im = fig.add_subplot(gs[row:row + 2, col])
            image_axes[key] = ax_im

        ax_main.set_xlabel('Wavelength (nm)', fontsize=18, labelpad=15)
        ax_main.set_ylabel('Intensity (a.u.)', fontsize=18, labelpad=15)

        ax_main.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
        ax_main.tick_params(axis='both', which='minor', length=4, width=1)

        secax = ax_main.secondary_xaxis('top')
        secax.set_xlabel('Energy (eV)', fontsize=18, labelpad=15)
        secax.tick_params(axis='x', labelsize=14, length=6, width=1.5)

        wavelength_ticks = np.linspace(min(wavelengths), max(wavelengths), num=6)
        energy_ticks = np.linspace(wavelengths[0], wavelengths[-1], num=5)
        secax.set_xticks(energy_ticks)
        secax.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

        ax_main.grid(True)
        ax_main.grid(True, which='both', color='black', linestyle='--', linewidth=0.3)
        for spine in ax_main.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax_main.axvspan(950, 1000, color='green', alpha=0.15, label="950-1000 nm")
        ax_main.axvspan(870, 940, color='blue', alpha=0.15, label="870-940 nm")

        ax_main.legend()

        return ax_main, image_axes, colors

    def __setup_plotting_single_image(self, figsize, wavelengths):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(6, 7, figure=fig, wspace=0.1, hspace=0.2)
        ax_main = fig.add_subplot(gs[:, 0:4])
        ax_im = fig.add_subplot(gs[:, 4:])

        ax_main.set_xlabel('Wavelength (nm)', fontsize=18, labelpad=15)
        ax_main.set_ylabel('Intensity (a.u.)', fontsize=18, labelpad=15)

        ax_main.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
        ax_main.tick_params(axis='both', which='minor', length=4, width=1)

        secax = ax_main.secondary_xaxis('top')
        secax.set_xlabel('Energy (eV)', fontsize=18, labelpad=15)
        secax.tick_params(axis='x', labelsize=14, length=6, width=1.5)

        wavelength_ticks = np.linspace(min(wavelengths), max(wavelengths), num=6)
        energy_ticks = np.linspace(wavelengths[0], wavelengths[-1], num=5)
        secax.set_xticks(energy_ticks)
        secax.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

        ax_main.grid(True)
        ax_main.grid(True, which='both', color='black', linestyle='--', linewidth=0.3)
        for spine in ax_main.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax_main.axvspan(950, 1000, color='green', alpha=0.15, label="950-1000 nm")
        ax_main.axvspan(870, 940, color='blue', alpha=0.15, label="870-940 nm")

        ax_main.legend()

        return ax_main, ax_im

    def __setup_coordinations(self, key_coord, image_axes):
        coord_x = []
        coord_y = []
        for i in self.data_coordinates.keys():
            if i == 'ref':
                coord_x.append(key_coord[1])
                coord_y.append(key_coord[0])
                CondAns.__plot_image_with_rect(image_axes[self.ref], self.data_dict[self.ref].get_live_scan(),
                                               key_coord, self.ref)
            else:
                temp_coord = self.data_coordinates[i].get(key_coord)
                if temp_coord is None:
                    temp_coord = (0, 0)
                coord_x.append(temp_coord[1])
                coord_y.append(temp_coord[0])
                CondAns.__plot_image_with_rect(image_axes[i], self.data_dict[i].get_live_scan(),
                                               temp_coord, i)

        return coord_x, coord_y

    def __peak_fitting_auto(self, intensity, wavelengths, height, prominence, distance, max_peaks, fit_func):

        self.best_r2 = None
        self.best_result = None
        self.best_model = None
        self.params_fit = None
        self.best_fit = None

        peaks_indices, properties = find_peaks(intensity, height=height, prominence=prominence, distance=distance)
        peak_positions = wavelengths[peaks_indices]
        peak_heights = intensity[peaks_indices]

        sorted_idx = np.argsort(peak_heights)[::-1]
        sorted_idx = sorted_idx[:max_peaks]
        peak_positions = peak_positions[sorted_idx]
        peak_heights = peak_heights[sorted_idx]

        r2_list = []
        params_list = []
        models = []
        results = []

        model = ConstantModel(prefix='bkg_')
        params = model.make_params(bkg_c=0)
        flag = False
        for m, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
            prefix = f'g{m}_'
            gauss = fit_func(prefix=prefix)
            model += gauss
            params.update(gauss.make_params())
            params[f'{prefix}amplitude'].set(value=height, min=0)
            params[f'{prefix}center'].set(value=pos)
            params[f'{prefix}sigma'].set(value=1, min=0.1)

            result = model.fit(intensity, params, x=wavelengths)

            ss_total = np.sum((intensity - np.mean(intensity)) ** 2)
            ss_residual = np.sum(result.residual ** 2)
            r_squared = 1 - (ss_residual / ss_total)
            r2_list.append(r_squared)
            params_list.append(result.params)
            models.append(model)
            results.append(result)

            print(f"Using {m + 1} peak(s): R² = {r_squared:.4f}")

            if len(peak_positions) == 1:
                self.best_fit = m
                self.params_fit = params_list[-1]
                self.best_model = models[-1]
                self.best_result = results[-1]
                self.best_r2 = r2_list[-1]
                flag = True
                break

            # if len(r2_list) == len(peak_positions):
            r2_temp = sorted(r2_list, reverse=True)
            for i in range(len(r2_temp) - 1):
                if r2_temp[i] - r2_temp[i + 1] > 0.005:
                    index = r2_list.index(r2_temp[i])
                    self.best_fit = index
                    self.params_fit = params_list[index]
                    self.best_model = models[index]
                    self.best_result = results[index]
                    self.best_r2 = r2_list[index]
                    flag = True
                    if len(r2_list) == len(peak_positions):
                        break

        if flag == False:
            self.best_fit = 1
            self.params_fit = params_list[0]
            self.best_model = models[0]
            self.best_result = results[0]
            self.best_r2 = r2_list[0]

            # if r2_list[-1] - r2_list[-2] < 0.007:
            #     self.best_fit = m
            #     self.params_fit = params_list[-2]
            #     self.best_model = models[-2]
            #     self.best_result = results[-2]
            #     self.best_r2 = r2_list[-2]
            # else:
            #     self.best_fit = m + 1
            #     self.params_fit = params_list[-1]
            #     self.best_model = models[-1]
            #     self.best_result = results[-1]
            #     self.best_r2 = r2_list[-1]

            # if len(r2_list) > 1 and r2_list[-1] - r2_list[-2] < 0.007 and r2_list[-2] > 0.99:
            #     self.best_fit = m
            #     self.params_fit = params_list[-2]
            #     self.best_model = models[-2]
            #     self.best_result = results[-2]
            #     self.best_r2 = r2_list[-2]
            #     # break

    def interactive_peak_fit(self, exp_key, start_row=0, start_col=0):
        """
        Interactive peak fitting with Prev/Next + direct-entry for row & col,
        plus a button to fit all spectra, store and list those below an R² threshold,
        with a progress bar for the Fit All operation, and export fit data to Excel.
        Optionally exclude low-R² spectra from the exported file.
        """
        from ipywidgets import IntProgress
        import pandas as pd
        from IPython.display import display, FileLink

        exp_dropdown = Dropdown(
            options=list(self.data_dict.keys()),
            value=exp_key,
            description='Dataset:',
            layout=Layout(align_items='center', margin='0 10px')
        )
        # pull out data
        data = self.data_dict[exp_key].get_numpy_spectra()
        wavelengths = self.data_dict[exp_key].get_wavelengths()
        n_rows, n_cols = data.shape[0], data.shape[1]
        image_scan = self.data_dict[exp_key].get_live_scan()

        # init storage
        self.last_low_r2 = []
        self.last_fit_results = []

        # state
        row, col = start_row, start_col
        max_row, max_col = data.shape[0] - 1, data.shape[1] - 1

        desc_w = '80px'

        field_layout = Layout(width='200px', height='40px')

        row_entry = BoundedIntText(
            value=row,
            min=0,
            max=max_row,
            description='Row:',
            style={'description_width': desc_w},
            layout=field_layout
        )

        col_entry = BoundedIntText(
            value=col,
            min=0,
            max=max_col,
            description='Col:',
            style={'description_width': desc_w},
            layout=field_layout
        )

        # nav buttons
        btn_prev_row = Button(description='←', tooltip='Previous row', layout=Layout(width='50px'))
        btn_next_row = Button(description='→', tooltip='Next row', layout=Layout(width='50px'))
        btn_prev_col = Button(description='←', tooltip='Previous col', layout=Layout(width='50px'))
        btn_next_col = Button(description='→', tooltip='Next col', layout=Layout(width='50px'))

        out = Output(layout=Layout(border='1px solid gray'))

        w_height = FloatSlider(value=0.1, min=0.0, max=np.max(data), step=0.01, description='height',
                               continuous_update=False)
        w_prominence = FloatSlider(value=0.1, min=0.0, max=20, step=0.01, description='prominence',
                                   continuous_update=False)
        w_distance = FloatSlider(value=1.0, min=0.0, max=40, step=1.0, description='distance', continuous_update=False)
        w_max_peaks = IntSlider(value=3, min=1, max=10, step=1, description='max_peaks', continuous_update=False)
        w_fit_func = Dropdown(
            options=[('Gaussian', GaussianModel), ('Lorentzian', LorentzianModel), ('Voigt', VoigtModel)],
            description='fit_func')
        btn_fit = Button(description='Fit Peaks', button_style='primary')

        # Fit All controls
        threshold_entry = BoundedFloatText(value=0.9, min=0.0, max=1.0, step=0.01, description='R² thresh:')
        include_low_chk = Checkbox(value=True, description='Include low R² in Excel')
        btn_fit_all = Button(description='Fit All', button_style='warning')

        def update_display(change=None):
            nonlocal row, col
            row, col = row_entry.value, col_entry.value
            intensity = data[row, col]
            with out:
                out.clear_output(wait=True)
                self.__peak_fitting_auto(intensity=intensity, wavelengths=wavelengths,
                                         height=w_height.value, prominence=w_prominence.value,
                                         distance=w_distance.value, max_peaks=w_max_peaks.value,
                                         fit_func=w_fit_func.value)
                # 2) create side‐by‐side plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # — left: spectrum + fit
                ax1.plot(wavelengths, intensity, label='data', markersize=3)
                best = self.best_model.eval(x=wavelengths, params=self.params_fit)
                ax1.plot(wavelengths, best, '-', label=f'best fit (R²={self.best_r2:.3f})')
                names = []
                for name, comp in self.best_model.eval_components(x=wavelengths, params=self.params_fit).items():
                    names.append(name)
                    if name != 'bkg_':
                        ax1.plot(wavelengths, comp, '--', label=name)
                ax1.set(xlabel='Wavelength', ylabel='Intensity',
                        title=f'{exp_key} @ (row={row},col={col})')
                ax1.legend()

                if image_scan is not None:
                    ax2.imshow(image_scan, cmap='gray')
                    ax2.axis('off')
                    ax2.set_title('Sample map')
                    img_h, img_w = image_scan.shape[:2]
                    cell_w = img_w / n_cols
                    cell_h = img_h / n_rows
                    x0 = (col * cell_w) - (cell_w / 2)
                    y0 = (row * cell_h) - (cell_h / 2)
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x0, y0), cell_w, cell_h,
                                     linewidth=2, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                else:
                    ax2.text(0.5, 0.5, 'No map image\nprovided',
                             ha='center', va='center', fontsize=12)
                    ax2.axis('off')

                plt.tight_layout()
                plt.show()

                params = self.params_fit

                # decide the order you want suffixes printed in
                suffix_order = ['amplitude', 'center', 'sigma', 'fwhm']

                # 1) collect all the prefixes (e.g. 'g0', 'g1', 'bkg')
                prefixes = sorted({name.split('_', 1)[0] for name in params.keys()})

                print(f"Fit parameters for {exp_key} @ (row={row},col={col}):")
                print(f"  overall R² = {self.best_r2:.4f}")
                print("  detailed parameters:")

                for pfx in prefixes:
                    print(f"  {pfx}_:")
                    for suf in suffix_order:
                        full = f"{pfx}_{suf}"
                        if full in params:
                            par = params[full]
                            print(f"    {suf:<9} = {par.value:10.3f}  ± {par.stderr:.3f}" if par.stderr else
                                  f"    {suf:<9} = {par.value:10.3f}")
                    extras = sorted(
                        name.split('_', 1)[1]
                        for name in params.keys()
                        if name.startswith(pfx + '_') and name.split('_', 1)[1] not in suffix_order
                    )
                    for suf in extras:
                        par = params[f"{pfx}_{suf}"]
                        print(f"    {suf:<9} = {par.value:10.3f}")

        btn_fit.on_click(lambda _: update_display())

        def fit_all_callback(_):
            self.last_low_r2.clear()
            self.last_fit_results.clear()
            thresh = threshold_entry.value
            total = data.shape[0] * data.shape[1]
            progress = IntProgress(min=0, max=total, description='Fitting:')
            count = 0
            low_list = []
            with out:
                out.clear_output()
                display(progress)

                for r in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        try:
                            self.__peak_fitting_auto(intensity=data[r, c], wavelengths=wavelengths,
                                                     height=w_height.value, prominence=w_prominence.value,
                                                     distance=w_distance.value, max_peaks=w_max_peaks.value,
                                                     fit_func=w_fit_func.value)
                            params = self.params_fit.valuesdict()
                            rec = {'row': r, 'col': c, 'r2': self.best_r2, 'fit_func': w_fit_func.value}
                            rec.update(params)
                            self.last_fit_results.append(rec)
                            if self.best_r2 < thresh: low_list.append((r, c, self.best_r2))
                        except Exception as e:
                            print(f"Error fitting (row={r}, col={c}): {e}")
                        count += 1
                        progress.value = count
                self.last_low_r2 = low_list
                # prepare DataFrame
                df = pd.DataFrame(self.last_fit_results)
                # filter if excluding low
                if not include_low_chk.value:
                    df = df[df['r2'] >= thresh]
                path = f'fitting_results_{exp_key}.xlsx'
                df.to_excel(path, index=False)
                # print summary
                if low_list:
                    print(f"Spectra with R² below {thresh}:")
                    for r, c, r2 in low_list: print(f"  row={r},col={c},R²={r2:.3f}")
                else:
                    print(f"All spectra have R² ≥ {thresh}.")
                print(f"\nExcel file saved to '{path}'.")
                display(FileLink(path))

        btn_fit_all.on_click(fit_all_callback)

        # navigation
        def shift_row(d):
            row_entry.value = np.clip(row_entry.value + d, 0, max_row)

        def shift_col(d):
            col_entry.value = np.clip(col_entry.value + d, 0, max_col)

        def refresh_for_new_key(new_key):
            """Re-load all of the per-key variables and reset controls."""
            nonlocal data, wavelengths, image_scan, n_rows, n_cols, max_row, max_col
            data = self.data_dict[new_key].get_numpy_spectra()
            wavelengths = self.data_dict[new_key].get_wavelengths()
            image_scan = self.data_dict[new_key].get_live_scan()
            n_rows, n_cols = data.shape[:2]
            max_row, max_col = n_rows - 1, n_cols - 1

            # update widget limits and reset position
            row_entry.max = max_row
            col_entry.max = max_col
            row_entry.value = 0
            col_entry.value = 0

            update_display()  # redraw immediately on key change

            # observer on the dropdown

        def on_key_change(change):
            if change['name'] == 'value' and change['new'] != change['old']:
                refresh_for_new_key(change['new'])

        exp_dropdown.observe(on_key_change, names='value')

        btn_prev_row.on_click(lambda _: shift_row(-1));
        btn_next_row.on_click(lambda _: shift_row(1))
        btn_prev_col.on_click(lambda _: shift_col(-1));
        btn_next_col.on_click(lambda _: shift_col(1))
        row_entry.observe(update_display, names='value');
        col_entry.observe(update_display, names='value')

        # # layout
        # row_ctrl = HBox([row_entry, btn_prev_row, btn_next_row], layout=Layout(align_items='center', margin='0 10px'))
        # col_ctrl = HBox([col_entry, btn_prev_col, btn_next_col], layout=Layout(align_items='center'))
        # ctrl = VBox([
        #     HBox([row_ctrl, col_ctrl]),
        #     HBox([w_height, w_prominence], layout=Layout(margin='10px 0')),
        #     HBox([w_distance, w_max_peaks, w_fit_func], layout=Layout(margin='10px 0')),
        #     HBox([btn_fit, threshold_entry, include_low_chk, btn_fit_all], layout=Layout(margin='10px 0'))
        # ])
        # display(VBox([ctrl, out]))
        # update_display()
        # now build the UI
        # controls = HBox([
        #     exp_dropdown,
        #     HBox([btn_prev_row, row_entry, btn_next_row]),
        #     HBox([btn_prev_col, col_entry, btn_next_col])
        # ])
        # fit_controls = HBox([w_height, w_prominence, w_distance, w_max_peaks, w_fit_func, btn_fit])
        # all_controls = HBox([threshold_entry, include_low_chk, btn_fit_all])
        #
        # display(VBox([controls, fit_controls, all_controls, out]))
        #

        row1 = exp_dropdown

        # Row 2: row/col navigator
        row_controls = HBox(
            [btn_prev_row, row_entry, btn_next_row],
            layout=Layout(
                display='flex',
                flex_flow='row nowrap',
                align_items='center',
                gap='5px'  # small CSS gap between all children
            )
        )

        col_controls = HBox(
            [btn_prev_col, col_entry, btn_next_col],
            layout=Layout(
                display='flex',
                flex_flow='row nowrap',
                align_items='center',
                gap='5px'  # small CSS gap between all children
            )
        )

        # Row 3: fit‐parameter sliders
        row5 = HBox([
            w_height, w_prominence, w_distance
        ], layout=Layout(margin='10px 10px', flex_flow='row wrap'))

        row3 = HBox([
            w_max_peaks, w_fit_func, btn_fit
        ], layout=Layout(margin='10px 10px', flex_flow='row wrap'))

        # Row 4: action buttons
        row4 = HBox([
            threshold_entry, include_low_chk, btn_fit_all
        ], layout=Layout(margin='10px 10px'))

        # === display all four rows + output below ===
        display(VBox([
            row1,
            row_controls,
            col_controls,
            row5,
            row3,
            row4,
            out
        ], layout=Layout(spacing='15px')))
        update_display()

    def interactive_peak_fit_manual(self, exp_key, start_row=0, start_col=0):
        """
        Interactive manual peak fitting with Prev/Next + direct-entry for row & col,
        plus controls to add/remove individual peak guesses (amplitude, center, sigma, function),
        collect them into a dict, call __peak_fitting_manual, and select & fit an ROI
        (square option available). Saves Excel of results for the ROI.
        """
        from ipywidgets import (
            FloatText, Dropdown, Label, HBox, VBox, Layout, Button, BoundedIntText,
            Checkbox, BoundedFloatText, IntProgress, Output
        )
        from IPython.display import display, FileLink
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import RectangleSelector
        import matplotlib as mpl

        field_layout = Layout(width='180px')
        label_style = {'description_width': '60px'}
        param_layout   = Layout(width='120px')   
        btn_layout = Layout(width='150px')
        hbox_gap       = {'gap': '4px', 'align_items': 'center'}

        self.last_low_r2 = []
        self.last_fit_results = []

        data = self.data_dict[exp_key].get_numpy_spectra()
        wavelengths = self.data_dict[exp_key].get_wavelengths()
        image_scan = self.data_dict[exp_key].get_live_scan()
        n_rows, n_cols = data.shape[:2]
        max_row, max_col = n_rows - 1, n_cols - 1

        row, col = start_row, start_col
        peak_widgets = []

        exp_dropdown = Dropdown(
            options=list(self.data_dict.keys()),
            value=exp_key,
            description='Dataset:',
            layout=Layout(margin='0 10px'),
            style=label_style
        )

        row_entry = BoundedIntText(
            value=row, min=0, max=max_row,
            description='Row:',
            layout=field_layout,
            style=label_style
        )
        col_entry = BoundedIntText(
            value=col, min=0, max=max_col,
            description='Col:',
            layout=field_layout,
            style=label_style
        )
        btn_prev_row = Button(description='←', layout=btn_layout)
        btn_next_row = Button(description='→', layout=btn_layout)
        btn_prev_col = Button(description='←', layout=btn_layout)
        btn_next_col = Button(description='→', layout=btn_layout)

        peaks_container = VBox(layout=Layout(border='1px solid lightgray', padding='5px'))
        btn_add_peak = Button(description='Add Peak', layout=btn_layout)
        btn_remove_peak = Button(description='Remove Peak', layout=btn_layout)
        btn_fit_manual = Button(description='Fit Manual', button_style='primary', layout=btn_layout)
        fit_bkg_cb = Checkbox(value=False, description='Fit Background Automatically', layout=Layout(margin='0 10px'))


        select_roi_btn = Button(description='Select ROI (draw)', layout=Layout(width='180px'))
        force_square_cb = Checkbox(value=False, description='Force square (expand short side)')
        roi_start_row = BoundedIntText(value=0, min=0, max=max_row, description='r start:', layout=Layout(width='120px'))
        roi_end_row   = BoundedIntText(value=0, min=0, max=max_row, description='r end:',   layout=Layout(width='120px'))
        roi_start_col = BoundedIntText(value=0, min=0, max=max_col, description='c start:', layout=Layout(width='120px'))
        roi_end_col   = BoundedIntText(value=0, min=0, max=max_col, description='c end:',   layout=Layout(width='120px'))
        btn_fit_selection = Button(description='Fit Selection', button_style='warning', layout=Layout(width='180px'))

        out = Output(layout=Layout(border='1px solid gray'))

        from ipywidgets import HTML

        def make_param(name, init, lo, hi):
            value_layout = Layout(width='80px', margin='0 2px')
            bound_layout = Layout(width='80px', margin='0 1px')
            col_layout   = Layout(display='flex', flex_flow='column', align_items='center', margin='0 4px')

            title = HTML(f'<b>{name}</b>', layout=Layout(margin='0 0 2px 0'))
            hdr = HBox([
                HTML('Val', layout=Layout(width=value_layout.width, margin='0 1px')),
                HTML('Min', layout=Layout(width=bound_layout.width, margin='0 1px')),
                HTML('Max', layout=Layout(width=bound_layout.width, margin='0 1px'))
            ], layout=Layout(gap='1px', margin='0 0 2px 0'))
            val  = FloatText(value=init, layout=value_layout)
            vmin = FloatText(value=lo,   layout=bound_layout)
            vmax = FloatText(value=hi,   layout=bound_layout)
            column = VBox([title, hdr, HBox([val, vmin, vmax], layout=Layout(gap='1px'))], layout=col_layout)
            return val, vmin, vmax, column

        def make_peak_row():
            row_layout   = Layout(display='flex', flex_flow='row', align_items='center', margin='4px 0')

            # Common parameters
            amp, amp_min, amp_max, amp_col = make_param('Amp',    1.0, 0.0, 0.0)
            cen, cen_min, cen_max, cen_col = make_param('Center', 0.0, 0.0, 0.0)
            sig, sig_min, sig_max, sig_col = make_param('Sigma',  1.0, 0.0, 0.0)

            # Extra parameter (gamma) – shown only when needed
            gam, gam_min, gam_max, gam_col = make_param('Gamma',  1.0, 0.0, 0.0)
            gam_col.layout.display = 'none'  # hidden by default

            func = Dropdown(
                options=[
                    ('Gauss', GaussianModel),
                    ('Lorentz', LorentzianModel),
                    ('Voigt', VoigtModel),
                    ('Lognormal', LognormalModel),
                    ('Exponential Gaussian', ExponentialGaussianModel),
                ],
                description='Func',
                layout=Layout(width='220px', margin='0 4px'),
                style={'description_width': '60px'}
            )
            exists = Checkbox(value=False, description='Always Exists')

            # toggle gamma visibility based on function
            def _on_func_change(change):
                if change['name'] == 'value':
                    cls = change['new']
                    need_gamma = (cls is ExponentialGaussianModel)
                    gam_col.layout.display = 'flex' if need_gamma else 'none'

            func.observe(_on_func_change, names='value')

            peak_widgets.append((
                amp, amp_min, amp_max,
                cen, cen_min, cen_max,
                sig, sig_min, sig_max,
                gam, gam_min, gam_max,
                func, exists
            ))

            row_box = HBox([amp_col, cen_col, sig_col, gam_col, func, exists], layout=row_layout)
            separator = HTML("<hr style='width:100%;margin:4px 0;color:lightgray;'>")

            return VBox([row_box, separator])

        # --- Add/Remove handlers ---
        def on_add_peak(_):
            widget_row = make_peak_row()
            peaks_container.children += (widget_row,)

        def on_remove_peak(_):
            if peak_widgets:
                peak_widgets.pop()
                peaks_container.children = peaks_container.children[:-1]

        threshold_entry = BoundedFloatText(value=0.005, min=0.0, max=1.0, description='R² thresh:')

        def _build_peak_list_from_widgets():
            plist = []
            for (
                amp_w, amp_min_w, amp_max_w,
                cen_w, cen_min_w, cen_max_w,
                sig_w, sig_min_w, sig_max_w,
                gam_w, gam_min_w, gam_max_w,
                func_w, exists_w
            ) in peak_widgets:
                entry = {
                    'amplitude':     amp_w.value,
                    'amplitude_min': None if amp_min_w.value == 0 else amp_min_w.value,
                    'amplitude_max': None if amp_max_w.value == 0 else amp_max_w.value,

                    'center':        cen_w.value,
                    'center_min':    None if cen_min_w.value == 0 else cen_min_w.value,
                    'center_max':    None if cen_max_w.value == 0 else cen_max_w.value,

                    'sigma':         sig_w.value,
                    'sigma_min':     None if sig_min_w.value == 0 else sig_min_w.value,
                    'sigma_max':     None if sig_max_w.value == 0 else sig_max_w.value,

                    'func':          func_w.value,
                    'exists':        exists_w.value
                }
                if func_w.value is ExponentialGaussianModel:
                    entry.update({
                        'gamma':      gam_w.value,
                        'gamma_min':  None if gam_min_w.value == 0 else gam_min_w.value,
                        'gamma_max':  None if gam_max_w.value == 0 else gam_max_w.value,
                    })
                plist.append(entry)
            return plist

        def update_display():
            nonlocal row, col, data, wavelengths
            row, col = row_entry.value, col_entry.value
            thresh = threshold_entry.value
            intensity = data[row, col]

            with out:
                out.clear_output(wait=True)
                peak_list = _build_peak_list_from_widgets()

                self.__peak_fitting_manual(
                    intensity=intensity,
                    wavelengths=wavelengths,
                    peak_params=peak_list,
                    r2_threshold=thresh, 
                    fit_bkg=fit_bkg_cb.value
                )

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                if fit_bkg_cb.value:
                    ax1.plot(wavelengths, intensity - self.params_fit['bkg_c'].value, label='data', markersize=3)
                    best = self.best_model.eval(x=wavelengths, params=self.params_fit)
                    ax1.plot(wavelengths, best - self.params_fit['bkg_c'].value, '-', label=f'best fit (R²={self.best_r2:.3f})')
                else:
                    ax1.plot(wavelengths, intensity, label='data', markersize=3)
                    best = self.best_model.eval(x=wavelengths, params=self.params_fit)
                    ax1.plot(wavelengths, best, '-', label=f'best fit (R²={self.best_r2:.3f})')

                for name, comp in self.best_model.eval_components(x=wavelengths, params=self.params_fit).items():
                    if name != 'bkg_':
                        ax1.plot(wavelengths, comp, '--', label=name)

                ax1.set(xlabel='Wavelength', ylabel='Intensity',
                        title=f'{exp_key} @ (row={row},col={col})')
                ax1.legend()

                if image_scan is not None:
                    ax2.imshow(image_scan, cmap='gray')
                    ax2.axis('off')
                    ax2.set_title('Sample map')
                    # cell rectangle
                    img_h, img_w = image_scan.shape[:2]
                    cell_w = img_w / n_cols
                    cell_h = img_h / n_rows
                    x0 = (col * cell_w) - (cell_w / 2)
                    y0 = (row * cell_h) - (cell_h / 2)
                    rect = Rectangle((x0, y0), cell_w, cell_h, linewidth=2, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                else:
                    ax2.text(0.5, 0.5, 'No map image\nprovided',
                            ha='center', va='center', fontsize=12)
                    ax2.axis('off')

                plt.tight_layout()
                plt.show()

                # print parameter table
                params = self.params_fit
                suffix_order = ['amplitude', 'center', 'sigma', 'gamma', 'fwhm']  # include gamma
                prefixes = sorted({name.split('_', 1)[0] for name in params.keys()})

                print(f"Fit parameters for {exp_key} @ (row={row},col={col}):")
                print(f"  overall R² = {self.best_r2:.4f}")
                print("  detailed parameters:")

                for pfx in prefixes:
                    print(f"  {pfx}_:")
                    for suf in suffix_order:
                        full = f"{pfx}_{suf}"
                        if full in params:
                            par = params[full]
                            print(
                                f"    {suf:<9} = {par.value:10.3f}  ± {par.stderr:.3f}"
                                if getattr(par, 'stderr', None)
                                else f"    {suf:<9} = {par.value:10.3f}"
                            )
                    extras = sorted(
                        name.split('_', 1)[1]
                        for name in params.keys()
                        if name.startswith(pfx + '_') and name.split('_', 1)[1] not in suffix_order
                    )
                    for suf in extras:
                        par = params[f"{pfx}_{suf}"]
                        print(f"    {suf:<9} = {par.value:10.3f}")

        # wire buttons
        btn_add_peak.on_click(lambda _: on_add_peak(None))
        btn_remove_peak.on_click(lambda _: on_remove_peak(None))
        btn_fit_manual.on_click(lambda _: update_display())

        btn_fit_all = Button(description='Fit All', button_style='warning')

        def fit_all_callback(_):
            self.last_low_r2.clear()
            self.last_fit_results.clear()
            thresh = threshold_entry.value
            total = data.shape[0] * data.shape[1]
            progress = IntProgress(min=0, max=total, description='Fitting:')
            count = 0
            low_list = []
            with out:
                out.clear_output()
                display(progress)
                peak_list = _build_peak_list_from_widgets()

                for r in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        try:
                            self.__peak_fitting_manual(
                                intensity=data[r, c],
                                wavelengths=wavelengths,
                                peak_params=peak_list,
                                r2_threshold=thresh, 
                                fit_bkg=fit_bkg_cb.value
                            )
                            params = self.params_fit.valuesdict()
                            print(f"Fitted (row={r}, col={c}), R²={self.best_r2:.4f}")
                            # print(params)
                            rec = {
                                'row': r, 'col': c, 'r2': self.best_r2,
                                'func_list': [peak['func'] for peak in peak_list]
                            }
                            rec.update(params)
                            print(f"Rec: {rec}")
                            self.last_fit_results.append(rec)
                            print(f"Appended record for (row={r}, col={c}) to results.")

                        except Exception as e:
                            print(f"Error fitting (row={r}, col={c}): {e}")
                        count += 1
                        progress.value = count

                self.last_low_r2 = low_list
                # print(f"Final fit results: {self.last_fit_results}")
                df = pd.DataFrame(self.last_fit_results)
                # print(df)
                path = f'fitting_results_{exp_key}.xlsx'
                df.to_excel(path, index=False)
                print(f"\nExcel file saved to '{path}'.")
                display(FileLink(path))

        btn_fit_all.on_click(fit_all_callback)

        # --- ROI selection machinery ---
        rect_selector = {'active': False, 'selector': None, 'visual_rect': None}

        def _pixelcoords_to_grid_indices(xmin, ymin, xmax, ymax):
            """
            Convert pixel-space rectangle (xmin,xmax in image x coords, ymin,ymax in image y coords)
            into integer row/col indices using the same cell_w/cell_h mapping as the display.
            Returns (r0, r1, c0, c1) inclusive.
            """
            img_h, img_w = image_scan.shape[:2]
            cell_w = img_w / n_cols
            cell_h = img_h / n_rows

            # convert to approximate center-based grid index (rounded)
            def x_to_col(xpix):
                # +0.5 offset then floor => nearest integer center mapping
                return int(np.clip(np.floor(xpix / cell_w + 0.5), 0, n_cols - 1))

            def y_to_row(ypix):
                return int(np.clip(np.floor(ypix / cell_h + 0.5), 0, n_rows - 1))

            # map min/max
            c0 = x_to_col(xmin)
            c1 = x_to_col(xmax)
            r0 = y_to_row(ymin)
            r1 = y_to_row(ymax)

            # ensure ordered
            rmin, rmax = min(r0, r1), max(r0, r1)
            cmin, cmax = min(c0, c1), max(c0, c1)
            return rmin, rmax, cmin, cmax

        def _on_select(eclick, erelease):
            # eclick/erelease are matplotlib mouse events with xdata,ydata in image coordinates
            if eclick.xdata is None or erelease.xdata is None:
                return
            xmin = min(eclick.xdata, erelease.xdata)
            xmax = max(eclick.xdata, erelease.xdata)
            ymin = min(eclick.ydata, erelease.ydata)
            ymax = max(eclick.ydata, erelease.ydata)

            r0, r1, c0, c1 = _pixelcoords_to_grid_indices(xmin, ymin, xmax, ymax)

            # Optionally force square by expanding shorter side
            if force_square_cb.value:
                nrows = r1 - r0 + 1
                ncols = c1 - c0 + 1
                if nrows > ncols:
                    extra = nrows - ncols
                    # expand equally left/right if possible
                    left_expand = extra // 2
                    right_expand = extra - left_expand
                    c0 = max(0, c0 - left_expand)
                    c1 = min(n_cols - 1, c1 + right_expand)
                elif ncols > nrows:
                    extra = ncols - nrows
                    top_expand = extra // 2
                    bottom_expand = extra - top_expand
                    r0 = max(0, r0 - top_expand)
                    r1 = min(n_rows - 1, r1 + bottom_expand)

            # update widgets
            roi_start_row.value, roi_end_row.value = int(r0), int(r1)
            roi_start_col.value, roi_end_col.value = int(c0), int(c1)

            # draw visual rectangle on last displayed axes (if any)
            with out:
                try:
                    # clear previous rect
                    fig = plt.gcf()
                    ax = fig.axes[-1] if fig.axes else None
                    if ax is None:
                        return
                    # remove old visual rect if present
                    if rect_selector['visual_rect'] is not None:
                        try:
                            rect_selector['visual_rect'].remove()
                        except Exception:
                            pass
                        rect_selector['visual_rect'] = None

                    img_h, img_w = image_scan.shape[:2]
                    cell_w = img_w / n_cols
                    cell_h = img_h / n_rows
                    # convert grid indices back to left/top pixel coordinates similar to display logic
                    x0 = (c0 * cell_w) - (cell_w / 2)
                    y0 = (r0 * cell_h) - (cell_h / 2)
                    width = (c1 - c0 + 1) * cell_w
                    height = (r1 - r0 + 1) * cell_h
                    rect = Rectangle((x0, y0), width, height, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    rect_selector['visual_rect'] = rect
                    plt.draw()
                except Exception:
                    pass

        def toggle_selector(btn):
            # activate/deactivate the RectangleSelector
            if image_scan is None:
                with out:
                    print("No image available to select ROI from.")
                return

            if not rect_selector['active']:
                # create new rectangle selector over the image axes
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.imshow(image_scan, cmap='gray')
                ax.set_title('Draw ROI: click-drag-release')
                ax.axis('off')
                selector = RectangleSelector(ax, _on_select, drawtype='box', useblit=True,
                                            button=[1],  # left click only
                                            minspanx=2, minspany=2, spancoords='pixels',
                                            interactive=True)
                rect_selector['selector'] = selector
                rect_selector['active'] = True

                # change button text
                select_roi_btn.description = 'Cancel ROI'
            else:
                # deactivate
                try:
                    sel = rect_selector.get('selector', None)
                    if sel is not None:
                        sel.disconnect_events()
                        sel.set_active(False)
                except Exception:
                    pass
                rect_selector['selector'] = None
                rect_selector['active'] = False
                select_roi_btn.description = 'Select ROI (draw)'

        select_roi_btn.on_click(lambda _: toggle_selector(None))

        def fit_selection_callback(_):
            # read ROI numeric boxes and iterate only that region
            r0, r1 = int(roi_start_row.value), int(roi_end_row.value)
            c0, c1 = int(roi_start_col.value), int(roi_end_col.value)
            # sanitize
            r0, r1 = max(0, min(r0, n_rows - 1)), max(0, min(r1, n_rows - 1))
            c0, c1 = max(0, min(c0, n_cols - 1)), max(0, min(c1, n_cols - 1))
            if r1 < r0:
                r0, r1 = r1, r0
            if c1 < c0:
                c0, c1 = c1, c0

            self.last_fit_results.clear()
            thresh = threshold_entry.value
            total = (r1 - r0 + 1) * (c1 - c0 + 1)
            progress = IntProgress(min=0, max=total, description='Fitting ROI:')
            count = 0
            peak_list = _build_peak_list_from_widgets()

            with out:
                out.clear_output()
                display(progress)
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        try:
                            self.__peak_fitting_manual(
                                intensity=data[rr, cc],
                                wavelengths=wavelengths,
                                peak_params=peak_list,
                                r2_threshold=thresh
                            )
                            params = self.params_fit.valuesdict()
                            rec = {
                                'row': rr, 'col': cc, 'r2': self.best_r2,
                                'func_list': [peak['func'] for peak in peak_list]
                            }
                            rec.update(params)
                            self.last_fit_results.append(rec)
                        except Exception as e:
                            print(f"Error fitting (row={rr}, col={cc}): {e}")
                        count += 1
                        progress.value = count

                df = pd.DataFrame(self.last_fit_results)
                path = f'fitting_results_{exp_key}_ROI_r{r0}-{r1}_c{c0}-{c1}.xlsx'
                df.to_excel(path, index=False)
                print(f"\nExcel file saved to '{path}'.")
                display(FileLink(path))

        btn_fit_selection.on_click(fit_selection_callback)

        # navigation logic
        def shift_row(d):
            row_entry.value = np.clip(row_entry.value + d, 0, max_row)

        def shift_col(d):
            col_entry.value = np.clip(col_entry.value + d, 0, max_col)

        btn_prev_row.on_click(lambda _: shift_row(-1))
        btn_next_row.on_click(lambda _: shift_row(1))
        btn_prev_col.on_click(lambda _: shift_col(-1))
        btn_next_col.on_click(lambda _: shift_col(1))

        row_entry.observe(lambda _: update_display(), names='value')
        col_entry.observe(lambda _: update_display(), names='value')

        def on_key_change(change):
            if change['name'] == 'value' and change['new'] != change['old']:
                nonlocal data, wavelengths, image_scan, n_rows, n_cols, max_row, max_col, exp_key
                exp_key = change['new']
                data = self.data_dict[change['new']].get_numpy_spectra()
                wavelengths = self.data_dict[change['new']].get_wavelengths()
                image_scan = self.data_dict[change['new']].get_live_scan()
                n_rows, n_cols = data.shape[:2]
                max_row, max_col = n_rows - 1, n_cols - 1
                row_entry.max, col_entry.max = max_row, max_col
                row_entry.value, col_entry.value = 0, 0
                # also update ROI widgets limits
                
                roi_start_row.max = max_row
                roi_end_row.max = max_row
                roi_start_col.max = max_col
                roi_end_col.max = max_col
                update_display()

        exp_dropdown.observe(on_key_change, names='value')

        # assemble the top-level layout
        nav_row = HBox([btn_prev_row, row_entry, btn_next_row], layout=Layout(gap='5px'))
        nav_col = HBox([btn_prev_col, col_entry, btn_next_col], layout=Layout(gap='5px'))
        peak_ctrl = HBox([btn_add_peak, btn_remove_peak, btn_fit_manual, threshold_entry, btn_fit_all, fit_bkg_cb],
                        layout=Layout(margin='10px 0', gap='10px'))

        roi_ctrl_row = HBox([select_roi_btn, force_square_cb, btn_fit_selection], layout=Layout(gap='10px'))
        roi_numeric_row = HBox([roi_start_row, roi_end_row, roi_start_col, roi_end_col],
                            layout=Layout(margin='6px 0', gap='6px'))

        display(VBox([
            exp_dropdown,
            nav_row,
            nav_col,
            peak_ctrl,
            peaks_container,
            roi_ctrl_row,
            roi_numeric_row,
            out
        ], layout=Layout(spacing='10px')))


    def interactive_plot_fitted_data_all(self, addr, start_row=0, start_col=0):

        data = self.data_dict[self.ref].get_numpy_spectra()
        wavelengths = self.data_dict[self.ref].get_wavelengths()
        n_rows, n_cols = data.shape[0], data.shape[1]
        image_scan = self.data_dict[self.ref].get_live_scan()

        row, col = start_row, start_col
        max_row, max_col = data.shape[0] - 1, data.shape[1] - 1

        desc_w = '80px'

        field_layout = Layout(width='200px', height='40px')

        row_entry = BoundedIntText(
            value=row,
            min=0,
            max=max_row,
            description='Row:',
            style={'description_width': desc_w},
            layout=field_layout
        )

        col_entry = BoundedIntText(
            value=col,
            min=0,
            max=max_col,
            description='Col:',
            style={'description_width': desc_w},
            layout=field_layout
        )

        # nav buttons
        btn_prev_row = Button(description='←', tooltip='Previous row', layout=Layout(width='50px'))
        btn_next_row = Button(description='→', tooltip='Next row', layout=Layout(width='50px'))
        btn_prev_col = Button(description='←', tooltip='Previous col', layout=Layout(width='50px'))
        btn_next_col = Button(description='→', tooltip='Next col', layout=Layout(width='50px'))

        out = Output(layout=Layout(border='1px solid gray'))

        def shift_row(d):
            row_entry.value = np.clip(row_entry.value + d, 0, max_row)

        def shift_col(d):
            col_entry.value = np.clip(col_entry.value + d, 0, max_col)

        def update_display(change=None):
            nonlocal row, col
            peaks_data = dict()
            row, col = row_entry.value, col_entry.value
            intensity = data[row, col]
            row_a, col_a = row, col
            with out:
                out.clear_output(wait=True)
                cmap = plt.get_cmap('gnuplot')
                colors = cmap(np.linspace(0, 1, len(self.data_dict.keys())))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                for idx, key in enumerate(self.data_coordinates):
                    if key == 'ref':
                        i = self.ref
                        peaks_data[i] = []
                        row_a, col_a = row, col
                    else:
                        i = key
                        peaks_data[key] = []
                        print(f"Processing {i} @ (row={row}, col={col})")
                        row_a, col_a = self.data_coordinates[i][(row, col)]

                    x = self.data_dict[i].get_wavelengths()[::-1]
                    y = self.data_dict[i].get_numpy_spectra()[row_a, col_a][::-1]

                    res = self.__remodel_fitted_data(f'./{addr}/fitting_results_{i}.xlsx', int(row), int(col), x, y,
                                                     peak_type='lorentzian')
                    ax1.plot(x, res.best_fit,
                             color=colors[idx],
                             label=i)
                    j = 0
                    while True:
                        # amp_name = f"p{i}_height"
                        cen_name = f"p{j}_center"
                        if cen_name not in res.params:
                            break
                        # amp = res.params[amp_name].value
                        cen = res.params[cen_name].value
                        if cen < x[0] or cen > x[-1]:
                            j += 1
                            continue
                        
                        idx_cen = np.abs(x - cen).argmin()
                        peaks_data[i].append((cen, res.best_fit[idx_cen]))
                        j += 1

                    # for idx, (pos, height) in enumerate(peaks, 1):
                    #     print(f"Peak {idx}: x = {pos:.3f}, height = {height:.3f}")
                    # print(f'{i} : {max(res.best_fit)}')
                    # print(f'{i} : {x[np.argmax(res.best_fit)]}')
                    # y = res.best_fit
                    # peaks, props = find_peaks(y, height=0)
                    # peak_heights   = props["peak_heights"]
                    # peak_positions = x[peaks]
                    # for idx, (pos, height) in enumerate(zip(peak_positions, peak_heights), start=1):
                    #     # print(f"{i} – Peak {idx}: x = {pos:.3f}, height = {height:.3f}")
                    #     peaks_data[i].append((pos, height))
                   


                if image_scan is not None:
                    ax2.imshow(image_scan, cmap='gray')
                    ax2.axis('off')
                    ax2.set_title('Sample map')
                    img_h, img_w = image_scan.shape[:2]
                    cell_w = img_w / n_cols
                    cell_h = img_h / n_rows
                    x0 = (col * cell_w) - (cell_w / 2)
                    y0 = (row * cell_h) - (cell_h / 2)
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x0, y0), cell_w, cell_h,
                                     linewidth=2, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                else:
                    ax2.text(0.5, 0.5, 'No map image\nprovided',
                             ha='center', va='center', fontsize=12)
                    ax2.axis('off')

                plt.tight_layout()
                ax1.legend(title='Curve index')
                plt.show()
                def key_value_in_thousands(key):
                    # num_str = key.split("_")[-1]
                    # if num_str.endswith("NA"):
                    #     return int(num_str.rstrip("NA"))
                    # elif num_str.endswith("PA"):
                    #     return int(num_str.rstrip("PA")) / 1000
                    num_str = key.split("-")[-1]
                    return int(num_str.rstrip("K"))

                sorted_items = sorted(peaks_data.items(), key=lambda kv: key_value_in_thousands(kv[0]))
                peaks_data = {k: v for k, v in sorted_items}
                self.temp_str = ''
                for k,v in peaks_data.items():
                    if v:
                        print(f"Peaks for {k} @ (row={row}, col={col}):")
                        self.temp_str += f"Peaks for {k} @ (row={row}, col={col}):\n"
                        for idx, (pos, height) in enumerate(v, start=1):
                            print(f"  Peak {idx}: x = {pos:.3f}, height = {height:.3f}")
                            self.temp_str += f"  Peak {idx}: x = {pos:.3f}, height = {height:.3f}\n"
                    else:
                        print(f"No peaks found for {k} @ (row={row}, col={col})")         

        btn_prev_row.on_click(lambda _: shift_row(-1))
        btn_next_row.on_click(lambda _: shift_row(1))
        btn_prev_col.on_click(lambda _: shift_col(-1))
        btn_next_col.on_click(lambda _: shift_col(1))
        row_entry.observe(update_display, names='value')
        col_entry.observe(update_display, names='value')

        row_controls = HBox(
            [btn_prev_row, row_entry, btn_next_row],
            layout=Layout(
                display='flex',
                flex_flow='row nowrap',
                align_items='center',
                gap='5px'
            )
        )

        col_controls = HBox(
            [btn_prev_col, col_entry, btn_next_col],
            layout=Layout(
                display='flex',
                flex_flow='row nowrap',
                align_items='center',
                gap='5px'
            )
        )
        display(VBox([
            row_controls,
            col_controls,
            out
        ], layout=Layout(spacing='15px')))
        update_display()

    def interactive_compare_conditions(self, start_row=0, start_col=0, ordered_conditions=None, wlim=None):
        """
        Interactive comparison of one pixel across conditions with:
        - Waterfall (top-left) + Heatmap (bottom-left)
        - Thumbnail grid of condition maps (right) with matching-colored markers
        - Manual pixel overrides per condition
        - Natural ordering (3keV,4keV,...) unless you set your own
        """
        import re, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
        from ipywidgets import (
            BoundedIntText, Button, Dropdown, SelectMultiple, FloatText,
            FloatSlider, Checkbox, HBox, VBox, Layout, Output
        )
        from matplotlib.patches import Rectangle
        from IPython.display import display
        from math import ceil

        # --------------------- data setup ---------------------
        ref_key = self.ref
        data_ref = self.data_dict[ref_key].get_numpy_spectra()
        wavelengths_ref = self.data_dict[ref_key].get_wavelengths()
        image_ref = self.data_dict[ref_key].get_live_scan()
        n_rows, n_cols = data_ref.shape[:2]

        def _natkey(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

        keys_all = list(self.data_dict.keys())
        if ordered_conditions:
            ordered_keys = [k for k in ordered_conditions if k in keys_all]
            ordered_keys += [k for k in keys_all if k not in ordered_keys]
        else:
            ordered_keys = sorted(keys_all, key=_natkey)

        # --------------------- widgets ------------------------
        w_row = BoundedIntText(value=start_row, min=0, max=n_rows-1, description='Ref Row:', layout=Layout(width='180px'))
        w_col = BoundedIntText(value=start_col, min=0, max=n_cols-1, description='Ref Col:', layout=Layout(width='180px'))

        btn_prev_row = Button(description='↑', layout=Layout(width='36px'))
        btn_next_row = Button(description='↓', layout=Layout(width='36px'))
        btn_prev_col = Button(description='←', layout=Layout(width='36px'))
        btn_next_col = Button(description='→', layout=Layout(width='36px'))

        # w_order_sel = SelectMultiple(
        #     options=ordered_keys, value=tuple(ordered_keys),
        #     description='Conditions:', layout=Layout(width='320px', height='140px')
        # )

        w_norm   = Dropdown(options=[('None','none'),('Max','max'),('Area','area'),('Z-score','zscore')],
                            value='none', description='Normalize:')
        w_offset = FloatText(value=0.0, description='Offset:')
        w_cmap   = Dropdown(options=['viridis','plasma','inferno','magma','cividis','turbo'],
                            value='turbo', description='Colormap:')
        w_alpha  = FloatSlider(value=0.9, min=0.1, max=1.0, step=0.05, description='Alpha:')
        w_lw     = FloatSlider(value=1.5, min=0.5, max=4.0, step=0.1, description='Linewidth:')
        w_logy   = Checkbox(value=False, description='Log Y')

        # Layout controls
        w_show_all_maps = Checkbox(value=True, description='Show maps grid')
        w_maps_cols     = BoundedIntText(value=4, min=1, max=8, description='Thumbs/row:', layout=Layout(width='150px'))
        w_fig_w         = FloatSlider(value=14, min=10, max=24, step=0.5, description='Fig width')
        w_thumb_hboost  = FloatSlider(value=2.4, min=1.2, max=3.5, step=0.1, description='Thumb row height')

        out = Output(layout=Layout(border='1px solid #ccc', padding='6px'))

        # --- Manual correction storage
        manual_override = {cond: {'row': None, 'col': None} for cond in ordered_keys}
        override_widgets = {
            cond: (BoundedIntText(value=0, min=0, max=n_rows-1, description=f'{cond} row:', style={'description_width': '150px'}, layout=Layout(width='300px')),
                BoundedIntText(value=0, min=0, max=n_cols-1, description=f'{cond} col:', style={'description_width': '150px'}, layout=Layout(width='300px')))
            for cond in ordered_keys
        }
        btn_apply_overrides = Button(description='Apply Overrides', button_style='info')
        btn_reset_overrides = Button(description='Reset Overrides', button_style='warning')

        # --------------------- helpers -------------------
        def _normalize(spec, mode):
            if mode=='none':   return spec
            if mode=='max':    return spec / (np.max(spec) + 1e-12)
            if mode=='area':   return spec / (np.sum(spec) + 1e-12)
            if mode=='zscore': return (spec - np.mean(spec)) / (np.std(spec) + 1e-12)
            return spec

        def shift_row(d):
            w_row.value = int(np.clip(w_row.value + d, 0, n_rows-1))
            manual_override = {cond: {'row': None, 'col': None} for cond in ordered_keys}
        def shift_col(d):
            w_col.value = int(np.clip(w_col.value + d, 0, n_cols-1))
            manual_override = {cond: {'row': None, 'col': None} for cond in ordered_keys}
        def apply_overrides(_):
            for cond, (w_r, w_c) in override_widgets.items():
                if cond == self.ref:
                    continue  # skip ref condition
                manual_override[cond]['row'] = int(w_r.value)
                manual_override[cond]['col'] = int(w_c.value)
                self.data_coordinates[cond][(w_row.value, w_col.value)] = (manual_override[cond]['row'],
                                                                        manual_override[cond]['col'])
            update_display()
            
        def reset_overrides(_):
            for cond in ordered_keys:
                manual_override[cond] = {'row': None, 'col': None}
            update_display()

        btn_apply_overrides.on_click(apply_overrides)

        last_fig = {'fig': None}  # (optional) keep handle if you later add an Export button

        # --------------------- plotting -----------------------
        def update_display(_=None):
            row, col = w_row.value, w_col.value
            cond_order = ordered_keys
            cmap = mpl.cm.get_cmap(w_cmap.value)
            N = len(cond_order)
            colors = cmap(np.linspace(0, 1, max(N, 2)))
            with out:
                out.clear_output(wait=True)

                # ---- collect spectra in chosen order
                specs, labels, wls = [], [], []
                for cond in cond_order:
                    if cond not in self.data_dict:
                        continue
                    data = self.data_dict[cond].get_numpy_spectra()
                    wl   = self.data_dict[cond].get_wavelengths()

                    # choose pixel: override > mapping > ref
                    if manual_override[cond]['row'] is not None:
                        r = manual_override[cond]['row']
                        c = manual_override[cond]['col']
                    elif cond == ref_key:
                        r, c = row, col
                    else:
                        r, c = self.data_coordinates.get(cond, {}).get((row, col), (None, None))
                        if r is None:
                            continue
                    override_widgets[cond][0].value = r
                    override_widgets[cond][1].value = c
                    spec = _normalize(data[r, c].astype(float), w_norm.value)
                    specs.append(spec); labels.append(cond); wls.append(wl)

                # ---- figure size scales with #map rows
                cols = max(1, int(w_maps_cols.value))
                rows = int(ceil(len(labels) / cols)) if (w_show_all_maps.value and len(labels)>0) else 1
                fig_h = 6 + w_thumb_hboost.value * (rows - 1 if w_show_all_maps.value else 0)
                fig = plt.figure(constrained_layout=True, figsize=(w_fig_w.value, fig_h))
                last_fig['fig'] = fig

                # parent grid: 2 rows × 2 cols
                gs = fig.add_gridspec(
                    nrows=2, ncols=2,
                    width_ratios=[2.6, 2.0], height_ratios=[1, 1],
                    wspace=0.02, hspace=0.02
                )

                # left: waterfall (top) + heatmap (bottom)
                ax_wf = fig.add_subplot(gs[0, 0])
                ax_hm = fig.add_subplot(gs[0, 1])

                # right: either ref map only, or a grid of maps
                if w_show_all_maps.value and len(labels) > 0:
                    maps_gs = gs[1, :].subgridspec(rows, cols, wspace=0.08, hspace=0.12)
                    map_axes = [fig.add_subplot(maps_gs[i, j]) for i in range(rows) for j in range(cols)]
                else:
                    ax_ref = fig.add_subplot(gs[1, :])
                    map_axes = [ax_ref]

                # ---- Waterfall
                off = float(w_offset.value)
                if off == 0 and specs:
                    off = 0.9 * float(np.median([np.max(s) for s in specs]))
                for i, (cond, spec) in enumerate(zip(labels, specs)):
                    ax_wf.plot(wls[i], spec + i * off, color=colors[i % len(colors)],
                            lw=w_lw.value, alpha=w_alpha.value, label=cond)
                ax_wf.set_xlabel('Wavelength'); ax_wf.set_ylabel('Intensity + offset×k')
                if w_logy.value: ax_wf.set_yscale('log')
                ax_wf.legend(title='Condition', fontsize=8, loc='upper left', frameon=False)
                ax_wf.set_title(f'Pixel ({row},{col}) across {len(labels)} conditions')

                if len(specs) <= 1:
                    ax_hm.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                    ax_hm.axis('off')
                    return

                # ---- establish a common length across conditions and wavelengths
                # (be conservative: use the shortest among all specs and wls)
                min_len_specs = min(s.shape[0] for s in specs)
                min_len_wls   = min(len(w) for w in wls)
                min_len       = min(min_len_specs, min_len_wls)

                lam = np.asarray(wls[0])[:min_len]  # reference wavelength axis (trimmed)
                trimmed_specs = [np.asarray(s)[:min_len] for s in specs]

                # ---- apply wavelength window, if provided
                if wlim is not None:
                    lo, hi = sorted(wlim)
                    mask = (lam >= lo) & (lam <= hi)
                    # If nothing falls in the mask, bail gracefully
                    if not np.any(mask):
                        ax_hm.text(0.5, 0.5, 'No data in selected wavelength range', ha='center', va='center')
                        ax_hm.axis('off')
                        return
                    lam = lam[mask]
                    trimmed_specs = [s[mask] for s in trimmed_specs]

                # ---- stack (conds × λ)
                hm = np.vstack(trimmed_specs)                  # shape: (n_conditions, n_lambda)
                hm_disp = hm[:, ::-1]                          # flip columns to keep your visual style

                # x-extent should match the flipped order (high→low if lam is ascending, or vice versa)
                x0, x1 = lam[-1], lam[0]                       # because we flipped columns
                y0, y1 = len(hm_disp), 0

                im = ax_hm.imshow(
                    hm_disp,
                    aspect='auto',
                    origin='upper',
                    cmap=w_cmap.value if hasattr(w_cmap, "value") else w_cmap,
                    extent=[x0, x1, y0, y1],
                )

                ax_hm.set_title('Heatmap (conds × λ)')
                ax_hm.set_xlabel('Wavelength')
                ax_hm.set_ylabel('Condition')
                ax_hm.set_yticks(np.arange(len(labels)) + 0.5)
                ax_hm.set_yticklabels(labels)

                # Colorbar (caller should pass fig if it needs to attach elsewhere; here we reuse axis)
                cbar = ax_hm.figure.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel('Scaled intensity', rotation=270, labelpad=10)

                # ---- Right side: maps
                def _draw_marker(ax, img, rr, cc, edge):
                    ax.imshow(img, cmap='gray'); ax.axis('off')
                    ih, iw = img.shape[:2]
                    # map spectra matrix shape to image pixels
                    sp = self.data_dict[cond].get_numpy_spectra()
                    cw, ch = iw / sp.shape[1], ih / sp.shape[0]
                    x0, y0 = cc * cw, rr * ch
                    ax.add_patch(Rectangle((x0, y0), cw, ch, linewidth=2, edgecolor=edge, facecolor='none'))

                if w_show_all_maps.value and len(labels) > 0:
                    for i, cond in enumerate(labels):
                        axm = map_axes[i]
                        img = self.data_dict[cond].get_live_scan()
                        if img is None:
                            axm.text(0.5, 0.5, 'No map', ha='center', va='center'); axm.axis('off'); axm.set_title(cond)
                            continue

                        # decide (r,c) for this cond
                        if manual_override.get(cond, {}).get('row') is not None:
                            rr, cc = manual_override[cond]['row'], manual_override[cond]['col']
                        elif cond == ref_key:
                            rr, cc = row, col
                        else:
                            rr, cc = self.data_coordinates.get(cond, {}).get((row, col), (None, None))
                            if rr is None:
                                axm.imshow(img, cmap='gray'); axm.axis('off'); axm.set_title(cond + ' (no coord)')
                                continue

                        _draw_marker(axm, img, rr, cc, colors[i % len(colors)])
                        axm.set_title(cond, fontsize=10)

                    # hide any unused cells
                    for j in range(len(labels), len(map_axes)):
                        map_axes[j].axis('off')
                else:
                    # only ref map
                    axm = map_axes[0]
                    if image_ref is not None:
                        ih, iw = image_ref.shape[:2]
                        cw, ch = iw / n_cols, ih / n_rows
                        x0, y0 = w_col.value * cw, w_row.value * ch
                        axm.imshow(image_ref, cmap='gray'); axm.axis('off')
                        axm.add_patch(Rectangle((x0, y0), cw, ch, linewidth=2, edgecolor='red', facecolor='none'))
                        axm.set_title(f'Ref map: {ref_key}')
                    else:
                        axm.text(0.5, 0.5, 'No ref-map image', ha='center', va='center'); axm.axis('off')

                plt.show()

                # override table (quick sanity)
                print("Manual override table (row, col):")
                for cond in sorted(keys_all, key=_natkey):
                    ov = manual_override[cond]
                    print(f"  {cond}: row={ov['row']}, col={ov['col']}")

        # --------------------- wiring -------------------------
        btn_prev_row.on_click(lambda _: shift_row(-1))
        btn_next_row.on_click(lambda _: shift_row(1))
        btn_prev_col.on_click(lambda _: shift_col(-1))
        btn_next_col.on_click(lambda _: shift_col(1))

        for w in (w_row, w_col, w_norm, w_offset, w_cmap, w_alpha, w_lw, w_logy,
                w_show_all_maps, w_maps_cols, w_fig_w, w_thumb_hboost):
            w.observe(update_display, names='value')
        btn_apply_overrides.on_click(apply_overrides)
        btn_reset_overrides.on_click(reset_overrides)

        # --------------------- layout -------------------------
        overrides_ui = [HBox([w_r, w_c], layout=Layout(gap='4px')) for (w_r, w_c) in override_widgets.values()]
        overrides_box = VBox([Button(description="--- Manual Pixel Overrides ---", disabled=True)] +
                            overrides_ui + [btn_apply_overrides, btn_reset_overrides])

        nav_row = HBox([btn_prev_row, w_row, btn_next_row], layout=Layout(gap='6px'))
        nav_col = HBox([btn_prev_col, w_col, btn_next_col], layout=Layout(gap='6px'))
        layout_box = HBox([w_show_all_maps, w_maps_cols, w_fig_w, w_thumb_hboost], layout=Layout(gap='10px'))

        opts = VBox([
            HBox([w_norm, w_offset, w_cmap, w_alpha, w_lw, w_logy], layout=Layout(flex_flow='row wrap', gap='8px')),
            layout_box
        ])

        display(VBox([nav_row, nav_col, opts, overrides_box, out]))
        update_display()


    def __peak_fitting_manual(self, intensity, wavelengths, peak_params, r2_threshold=0.005, fit_bkg = False):
        peak_params.sort(key=lambda p: (p['exists'], p['amplitude']), reverse=True)
        
        if fit_bkg:
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
            ss_total    = np.sum((intensity - intensity.mean())**2)
            ss_residual = np.sum(result.residual**2)
            r_squared   = 1 - ss_residual/ss_total

            if p.get('exists') or len(r2_list) == 0:
                r2_list.append(r_squared)
                params_list.append(result.params)
                models.append(composite_model)
                results.append(result)

                # print(f"Using {i+1} peak(s): R² = {r_squared:.4f}")

                self.best_fit    = i
                self.params_fit  = params_list[-1]
                self.best_model  = models[-1]
                self.best_result = results[-1]
                self.best_r2     = r2_list[-1]
            else:
                if r_squared - r2_list[-1] > r2_threshold:
                    r2_list.append(r_squared)
                    params_list.append(result.params)
                    models.append(composite_model)
                    results.append(result)

                    # print(f"Using {i+1} peak(s): R² = {r_squared:.4f}")

                    self.best_fit    = i
                    self.params_fit  = params_list[-1]
                    self.best_model  = models[-1]
                    self.best_result = results[-1]
                    self.best_r2     = r2_list[-1]
        
        # for j in range(i + 1, len(r2_list)):
        #     idx = r2_list.index(r2_list[j])
            

        
        # after looping, pick the first bump in R² drop >0.005
        # r2_sorted = sorted(r2_list, reverse=True)
        # for j in range(len(r2_sorted)-1):
        #     idx = r2_list.index(r2_sorted[j])
        #     if len(params_list[idx]) <= always_exists_number:
        #         self.best_fit    = idx
        #         self.params_fit  = params_list[idx]
        #         self.best_model  = models[idx]
        #         self.best_result = results[idx]
        #         self.best_r2     = r2_list[idx]
        #     elif len(params_list[idx]) > always_exists_number and r2_sorted[j] - r2_sorted[j+1] > r2_threshold:
        #         self.best_fit    = idx
        #         self.params_fit  = params_list[idx]
        #         self.best_model  = models[idx]
        #         self.best_result = results[idx]
        #         self.best_r2     = r2_list[idx]
        #         if len(r2_list) == len(peak_params):
        #             break
    def interactive_waterfall(self, exp_key, start_row=0, start_col=0, end_row=None, end_col=None, wlim=None):
        """
        Interactive waterfall plotter (NO fitting).
        Adds:
        - index→(row,col) mapping table & export
        - per-curve colormap + colorbar
        - highlight a spectrum; optional per-k label on curves
        - colored markers on the map matching the curve colors
        - ### NEW: "Max intensity vs pixel" panel (raw or post-normalization)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from ipywidgets import (
            Dropdown, BoundedIntText, IntSlider, FloatText, FloatSlider, Checkbox,
            ToggleButtons, Button, HBox, VBox, Layout
        )
        from IPython.display import display
        import matplotlib as mpl

        # ---------- helpers ----------
        def _bresenham_line(r0, c0, r1, c1):
            r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
            dr = abs(r1 - r0); dc = abs(c1 - c0)
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
                if e2 <  dr: err += dr; c += s_c
            return pts

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

        def _extract_spectra(coords, data):
            specs, valid = [], []
            n_r, n_c = data.shape[:2]
            for (r, c) in coords:
                if 0 <= r < n_r and 0 <= c < n_c:
                    specs.append(data[r, c])
                    valid.append((r, c))
            return np.asarray(specs), valid

        def _normalize(specs, mode):
            if mode == "none": return specs
            if mode == "max":
                denom = np.maximum(specs.max(axis=1, keepdims=True), 1e-12); return specs / denom
            if mode == "area":
                denom = np.maximum(specs.sum(axis=1, keepdims=True), 1e-12); return specs / denom
            if mode == "zscore":
                mu = specs.mean(axis=1, keepdims=True); sd = specs.std(axis=1, keepdims=True) + 1e-12
                return (specs - mu) / sd
            return specs

        # ---------- load data ----------
        exp_dropdown = Dropdown(
            options=list(self.data_dict.keys()), value=exp_key,
            description='Dataset:', layout=Layout(width='280px')
        )
        data = self.data_dict[exp_key].get_numpy_spectra()
        wavelengths = self.data_dict[exp_key].get_wavelengths()
        image_scan = self.data_dict[exp_key].get_live_scan()
        n_rows, n_cols, n_lam = data.shape

        if end_row is None: end_row = min(start_row + 3, n_rows - 1)
        if end_col is None: end_col = min(start_col + 3, n_cols - 1)

        # ---------- widgets ----------
        w_mode = ToggleButtons(options=[('Line', 'line'), ('Rectangle', 'rect')], description='Region:')
        w_order = ToggleButtons(options=[('Row-major', 'row-major'), ('Col-major', 'col-major'), ('Along line', 'along-line')],
                                description='Order:')
        w_r0 = BoundedIntText(value=int(start_row), min=0, max=n_rows-1, description='Start row:')
        w_c0 = BoundedIntText(value=int(start_col), min=0, max=n_cols-1, description='Start col:')
        w_r1 = BoundedIntText(value=int(end_row),   min=0, max=n_rows-1, description='End row:')
        w_c1 = BoundedIntText(value=int(end_col),   min=0, max=n_cols-1, description='End col:')

        w_stride = IntSlider(value=1, min=1, max=10, step=1, description='Stride(px):', continuous_update=False)
        w_norm = Dropdown(options=[('None','none'), ('Max','max'), ('Area','area'), ('Z-score','zscore')],
                        value='none', description='Normalize:')
        w_offset = FloatText(value=0.0, description='Offset (auto=0):')

        w_cmap = Dropdown(
            options=['viridis','plasma','inferno','magma','cividis','turbo'],
            value='viridis', description='Colormap:'
        )
        w_alpha = FloatSlider(value=0.9, min=0.05, max=1.0, step=0.05, description='Alpha:', continuous_update=False)
        w_lw = FloatSlider(value=1.2, min=0.5, max=3.0, step=0.1, description='Linewidth:', continuous_update=False)
        w_logy = Checkbox(value=False, description='Log Y')
        w_heatmap = Checkbox(value=True, description='Heatmap')
        w_label_every = IntSlider(value=0, min=0, max=50, step=1, description='Label every k (0=off):')
        w_highlight = IntSlider(value=0, min=0, max=0, step=1, description='Highlight idx:', continuous_update=False)

        # ### NEW: controls for the max-intensity plot
        w_maxplot = Checkbox(value=True, description='Max vs pixel')
        w_max_source = Dropdown(
            options=[('Raw spectra','raw'), ('After normalization','norm')],
            value='raw', description='Max of:'
        )
        w_logy_max = Checkbox(value=False, description='Log Y (max)')

        btn_plot = Button(description='Plot Waterfall', button_style='primary', layout=Layout(width='180px'))
        btn_export = Button(description='Export CSV/PNG', layout=Layout(width='180px'))
        out = Output(layout=Layout(border='1px solid #ccc', padding='6px'))

        # ---------- plotting ----------
        state = {'valid_coords': [], 'mapping_df': None}

        def do_plot(save_paths=False):
            with out:
                out.clear_output(wait=True)

                # coords
                r0, c0, r1, c1 = w_r0.value, w_c0.value, w_r1.value, w_c1.value
                mode = w_mode.value
                order = w_order.value
                if mode == 'line':
                    coords = _bresenham_line(r0, c0, r1, c1)
                    if w_stride.value > 1: coords = coords[::w_stride.value]
                    order = 'along-line'
                else:
                    coords = _rect_coords(r0, c0, r1, c1, stride=w_stride.value, order=order)

                specs_raw, valid = _extract_spectra(coords, data)
                if len(valid) == 0:
                    print("No valid pixels in selection."); return

                specs_raw = specs_raw.astype(float)
                specs_norm = _normalize(specs_raw, w_norm.value)
                specs = specs_norm  # used for waterfall drawing

                # offset heuristic
                off = w_offset.value
                if off == 0.0:
                    off = 0.9 * np.median(np.max(specs if w_norm.value!='none' else specs_raw, axis=1))
                    if not np.isfinite(off) or off <= 0: off = 1.0

                # color setup
                N = len(valid)
                cmap = mpl.cm.get_cmap(w_cmap.value)
                colors = cmap(np.linspace(0, 1, N))
                norm = mpl.colors.Normalize(vmin=0, vmax=N-1)
                sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

                # figure layout: waterfall + optional map + optional heatmap + ### NEW: optional max-plot
                ncols_fig = 1
                if image_scan is not None: ncols_fig += 1
                if w_heatmap.value: ncols_fig += 1
                if w_maxplot.value: ncols_fig += 1  # ### NEW

                fig, axes = plt.subplots(1, ncols_fig, figsize=(18, 4),
                                        gridspec_kw={'width_ratios': [2] + [1]*(ncols_fig-1)})
                if ncols_fig == 1: axes = [axes]
                ax_wf = axes[0]

                # plot curves
                hi = np.clip(w_highlight.value, 0, N-1)
                for i, y in enumerate(specs):
                    lw = w_lw.value * (1.9 if i == hi else 1.0)
                    ax_wf.plot(wavelengths, y + i*off, lw=lw, alpha=w_alpha.value, color=colors[i])

                ax_wf.set_xlabel('Wavelength'); ax_wf.set_ylabel('Intensity + k*offset')
                title_region = f"LINE {valid[0]}→{valid[-1]}" if mode == 'line' else f"RECT ({r0},{c0})–({r1},{c1})"
                ax_wf.set_title(f"Waterfall • {exp_dropdown.value} • {title_region} • N={N}")
                if w_logy.value: ax_wf.set_yscale('log')
                cbar = fig.colorbar(sm, ax=ax_wf, fraction=0.046, pad=0.02)
                cbar.set_label('Spectrum index')

                # optional per-k labels
                k = int(w_label_every.value)
                if k > 0:
                    for i, y in enumerate(specs):
                        if i % k == 0:
                            ax_wf.text(wavelengths[-1], y[-1] + i*off, f"{i}:{valid[i]}", fontsize=8,
                                    va='bottom', ha='right', color=colors[i])

                ax_idx = 1

                # map overlay with color-matched markers
                if image_scan is not None:
                    ax_map = axes[ax_idx]; ax_idx += 1
                    ax_map.imshow(image_scan, cmap='gray'); ax_map.axis('off'); ax_map.set_title('Sample map')
                    img_h, img_w = image_scan.shape[:2]
                    cell_w = img_w / n_cols; cell_h = img_h / n_rows
                    xs = [c*cell_w + cell_w/2 for (_, c) in valid]
                    ys = [r*cell_h + cell_h/2 for (r, _) in valid]
                    ax_map.scatter(xs, ys, s=18, c=np.arange(N), cmap=cmap, norm=norm, edgecolors='none')
                    ax_map.scatter([xs[hi]], [ys[hi]], s=60, facecolors='none', edgecolors='w', linewidths=1.5)

                # heatmap (optional)
                if w_heatmap.value:
                    ax_hm = axes[ax_idx]; ax_idx += 1
                    lam_full   = np.asarray(wavelengths)
                    specs_full = np.asarray(specs)  # post-normalization view
                    hm_specs = specs_full[::-1, :]
                    hm_specs = hm_specs[:, ::-1]
                    lam      = lam_full[::-1]

                    if wlim is not None:
                        lo, hi_w = sorted(wlim)
                        lo = max(lo, float(lam.min()))
                        hi_w = min(hi_w, float(lam.max()))
                        mask = (lam >= lo) & (lam <= hi_w)

                        if mask.sum() < 2:
                            ax_hm.clear()
                            ax_hm.text(0.5, 0.5, 'No wavelengths in selected range',
                                    ha='center', va='center', transform=ax_hm.transAxes)
                            ax_hm.axis('off')
                        else:
                            lam_plot = lam[mask]
                            hm_plot  = hm_specs[:, mask]
                            y0, y1 = (0, hm_plot.shape[0])
                            x0, x1 = (lam_plot.min(), lam_plot.max())
                            ax_hm.clear()
                            ax_hm.imshow(hm_plot, aspect='auto', origin='upper',
                                        extent=[x0, x1, y0, y1], cmap=w_cmap.value)
                            ax_hm.set_xlabel('Wavelength')
                            ax_hm.set_ylabel('Spectrum index')
                            ax_hm.set_title(f'Heatmap (λ ∈ [{lo:.2f}, {hi_w:.2f}])')
                    else:
                        y0, y1 = (0, hm_specs.shape[0])
                        x0, x1 = (lam.min(), lam.max())
                        ax_hm.clear()
                        ax_hm.imshow(hm_specs, aspect='auto', origin='upper',
                                    extent=[x0, x1, y0, y1], cmap=w_cmap.value)
                        ax_hm.set_xlabel('Wavelength')
                        ax_hm.set_ylabel('Spectrum index')
                        ax_hm.set_title('Heatmap (post-normalization)')

                # ### NEW: Max intensity vs pixel (optional)
                if w_maxplot.value:
                    ax_max = axes[ax_idx]; ax_idx += 1
                    max_raw = np.max(specs_raw, axis=1)
                    max_norm = np.max(specs_norm, axis=1)
                    max_vals = max_raw if w_max_source.value == 'raw' else max_norm

                    xs_idx = np.arange(N)
                    ax_max.plot(xs_idx, max_vals, lw=1.2, alpha=0.8)
                    ax_max.scatter(xs_idx, max_vals, s=18, c=np.arange(N), cmap=cmap, norm=norm, edgecolors='none')
                    ax_max.scatter([xs_idx[hi]], [max_vals[hi]], s=60, facecolors='none', edgecolors='k', linewidths=1.0)
                    ax_max.set_xlabel('Spectrum index (selection order)')
                    ax_max.set_ylabel('Max intensity' + (' (raw)' if w_max_source.value == 'raw' else ' (norm)'))
                    ax_max.set_title('Max intensity vs pixel')
                    if w_logy_max.value:
                        ax_max.set_yscale('log')

                # build mapping dataframe (for export & potential debugging)
                df_map = pd.DataFrame({
                    'index': np.arange(N),
                    'row': [rc[0] for rc in valid],
                    'col': [rc[1] for rc in valid],
                    'max_raw': np.max(specs_raw, axis=1),
                    'max_norm': np.max(specs_norm, axis=1)
                })
                state['valid_coords'] = valid
                state['mapping_df'] = df_map

                plt.tight_layout()
                plt.show()

                # export
                if save_paths:
                    lam_hdr = [f"{wl:.6f}" for wl in wavelengths]
                    df_specs = pd.DataFrame(specs_raw, columns=lam_hdr)  # export RAW spectra by default
                    df_map.to_csv(f"waterfall_{exp_dropdown.value}_{mode}_map.csv", index=False)
                    df_specs.to_csv(f"waterfall_{exp_dropdown.value}_{mode}_spectra.csv", index=False)

                    # lightweight export plot
                    png_path = f"waterfall_{exp_dropdown.value}_{mode}.png"
                    fig2 = plt.figure(figsize=(10, 4))
                    ax2 = fig2.add_subplot(111)
                    for i, y in enumerate(specs):
                        ax2.plot(wavelengths, y + i*off, lw=w_lw.value, alpha=w_alpha.value,
                                color=cmap(i/(N-1) if N>1 else 0.5))
                    ax2.set_xlabel('Wavelength'); ax2.set_ylabel('Intensity + k*offset')
                    ax2.set_title(f"Waterfall (export) • {title_region} • N={N}")
                    fig2.tight_layout()
                    fig2.savefig(png_path, dpi=200); plt.close(fig2)
                    print("Saved: ",
                        f"waterfall_{exp_dropdown.value}_{mode}_map.csv, ",
                        f"waterfall_{exp_dropdown.value}_{mode}_spectra.csv, ",
                        png_path)

        # ---------- events ----------
        def refresh_on_dataset(change):
            if change['name'] == 'value' and change['new'] != change['old']:
                nonlocal data, wavelengths, image_scan, n_rows, n_cols, n_lam
                key = change['new']
                data = self.data_dict[key].get_numpy_spectra()
                wavelengths = self.data_dict[key].get_wavelengths()
                image_scan = self.data_dict[key].get_live_scan()
                n_rows, n_cols, n_lam = data.shape
                for w in (w_r0, w_r1):
                    w.max = n_rows - 1; w.value = min(w.value, n_rows - 1)
                for w in (w_c0, w_c1):
                    w.max = n_cols - 1; w.value = min(w.value, n_cols - 1)
                w_highlight.value = 0

        exp_dropdown.observe(refresh_on_dataset, names='value')
        btn_plot.on_click(lambda _: do_plot(save_paths=False))
        btn_export.on_click(lambda _: do_plot(save_paths=True))

        # Layout
        row_sel = HBox([w_r0, w_c0, w_r1, w_c1, w_stride], layout=Layout(gap='8px', flex_flow='row wrap'))
        opts1 = HBox([w_mode, w_order, w_norm, w_offset], layout=Layout(gap='8px', flex_flow='row wrap'))
        opts2 = HBox([
            w_cmap, w_alpha, w_lw, w_logy, w_heatmap, w_label_every, w_highlight,
            # ### NEW controls appended at the end
            w_maxplot, w_max_source, w_logy_max
        ], layout=Layout(gap='8px', flex_flow='row wrap'))
        actions = HBox([btn_plot, btn_export], layout=Layout(gap='10px'))

        display(VBox([
            exp_dropdown,
            opts1,
            row_sel,
            opts2,
            actions,
            HBox([out], layout=Layout(gap='10px'))
        ], layout=Layout(spacing='10px')))

        # initial draw
        do_plot(save_paths=False)


    def interactive_peak_refine(self, exp_key, start_row=0, start_col=0):
        """
        Manual‐only peak fitting panel where each peak can be
        Gaussian or Lorentzian independently.
        """
        # 1) data
        data = self.data_dict[exp_key].get_numpy_spectra()
        x = self.data_dict[exp_key].get_wavelengths()
        max_row, max_col = data.shape[0] - 1, data.shape[1] - 1

        # 2) row/col controls
        row_entry = BoundedIntText(value=start_row, min=0, max=max_row, description='Row:')
        col_entry = BoundedIntText(value=start_col, min=0, max=max_col, description='Col:')
        btn_pr = Button(description='←', layout=Layout(width='50px'))
        btn_nr = Button(description='→', layout=Layout(width='50px'))
        btn_pc = Button(description='←', layout=Layout(width='50px'))
        btn_nc = Button(description='→', layout=Layout(width='50px'))
        btn_pr.on_click(lambda _: setattr(row_entry, 'value', max(0, row_entry.value - 1)))
        btn_nr.on_click(lambda _: setattr(row_entry, 'value', min(max_row, row_entry.value + 1)))
        btn_pc.on_click(lambda _: setattr(col_entry, 'value', max(0, col_entry.value - 1)))
        btn_nc.on_click(lambda _: setattr(col_entry, 'value', min(max_col, col_entry.value + 1)))
        row_ctrl = HBox([row_entry, btn_pr, btn_nr],
                        layout=Layout(align_items='center', margin='0 50px 0 0'))
        col_ctrl = HBox([col_entry, btn_pc, btn_nc],
                        layout=Layout(align_items='center'))

        # 3) manual‐peak container + add button
        peaks_box = VBox()
        btn_add = Button(description='Add Peak', button_style='info')

        def add_peak(_):
            c = FloatText(value=0.0, description='Center', layout=Layout(width='200px'))
            s = FloatText(value=1.0, description='Sigma', layout=Layout(width='200px'))
            h = FloatText(value=1.0, description='Height', layout=Layout(width='200px'))
            dd = Dropdown(options=['Gaussian', 'Lorentzian'],
                          value='Gaussian',
                          description='Func',
                          layout=Layout(width='500px'))
            rm = Button(description='✖', layout=Layout(width='30px'))
            row = HBox([c, s, h, dd, rm], layout=Layout(align_items='center', spacing='10px'))
            # remove callback
            rm.on_click(lambda __: setattr(
                peaks_box, 'children',
                tuple(w for w in peaks_box.children if w is not row)
            ))
            # append
            peaks_box.children = peaks_box.children + (row,)

        btn_add.on_click(add_peak)
        add_peak(None)  # start with one

        # 4) fit button + output
        btn_fit = Button(description='Fit with Manual Guesses', button_style='primary', layout=Layout(width='400px'))
        out = Output(layout=Layout(border='1px solid gray'))

        def on_fit(_):
            out.clear_output()
            with out:
                if not peaks_box.children:
                    print("⚠️ Add at least one peak before fitting.")
                    return
                r, c = row_entry.value, col_entry.value
                y = data[r, c]

                # build composite model from each row's Func
                composite = None
                for i, row_w in enumerate(peaks_box.children):
                    prefix = f'p{i}_'
                    func = row_w.children[3].value  # the Dropdown
                    ModelClass = GaussianModel if func == 'Gaussian' else LorentzianModel
                    m = ModelClass(prefix=prefix)
                    composite = m if composite is None else composite + m

                # make & seed parameters
                params = composite.make_params()
                for i, row_w in enumerate(peaks_box.children):
                    prefix = f'p{i}_'
                    params[f'{prefix}center'].value = row_w.children[0].value
                    params[f'{prefix}sigma'].value = row_w.children[1].value
                    params[f'{prefix}height'].value = row_w.children[2].value

                # fit & plot
                result = composite.fit(y, params, x=x)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x, y, 'o', ms=3, label='data')
                comps = result.eval_components(x=x)
                for name, comp in comps.items():
                    ax.plot(x, comp, '--', label=name)
                ax.plot(x, result.best_fit, '-', lw=2,
                        label=f'fit (R²={result.rsquared:.3f})')
                ax.set_xlabel('Wavelength')
                ax.set_ylabel('Intensity')
                ax.set_title(f'{exp_key} @ (row={r}, col={c})')
                ax.legend()
                plt.show()

        btn_fit.on_click(on_fit)

        # 5) display
        controls = VBox([
            HBox([row_ctrl, col_ctrl]),
            btn_add,
            peaks_box,
            HBox([btn_fit], layout=Layout(margin='10px 0')),
        ])
        display(VBox([controls, out]))

    def __remodel_fitted_data(self,
                              excel_path: str,
                              row: int,
                              col: int,
                              x: np.ndarray,
                              y: np.ndarray,
                              peak_type: str
                              ):

        df = pd.read_excel(excel_path)
        pix = df[(df['row'] == row) & (df['col'] == col)]
        if pix.empty:
            raise ValueError(f"No fit parameters for row={row}, col={col}")
        pix = pix.iloc[0]

        # 2) pick model class
        # model_cls = {
        #     'gaussian': GaussianModel,
        #     'lorentzian': LorentzianModel,
        #     'voigt': VoigtModel
        # }.get(peak_type.lower())
        # if model_cls is None:
        #     raise ValueError("peak_type must be 'Gaussian', 'Lorentzian' or 'Voigt'")

        composite = None
        params = None
        i = 0
        while True:
            prefix = f"p{i}_"
            amp_key, cen_key, sig_key = prefix + 'amplitude', prefix + 'center', prefix + 'sigma'
            
            if amp_key not in pix or np.isnan(pix[amp_key]):
                break
            
            if float(pix[cen_key]) < 850:
                model_cls = GaussianModel
            elif float(pix[cen_key]) > 960:
                model_cls = VoigtModel
            else:
                model_cls = LorentzianModel
                
            

            mod = model_cls(prefix=prefix)
            if composite is None:
                composite = mod
            else:
                composite += mod

            par = mod.make_params(
                amplitude=pix[amp_key],
                center=pix[cen_key],
                sigma=pix[sig_key]
            )
            if params is None:
                params = par
            else:
                params.update(par)

            i += 1

        # print(i)
        bkg_mod = ConstantModel(prefix='bkg_c')
        composite += bkg_mod
        bkg_par = bkg_mod.make_params(c=pix['bkg_c'])
        params.update(bkg_par)

        result = composite.fit(y, params, x=x)

        return result

    @staticmethod
    def visualize_pixel_similarity(image1, image2, coord1, coord2, patch1, patch2, ssim_value, window_size=11,
                                   title='Image 2'):
        """
        Visualizes the pixel comparison process, showing the images, patches, and SSIM value.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # First image
        axes[0].imshow(image1, cmap='gray', extent=[0, 64, 64, 0])
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        axes[0].plot(coord1[1], coord1[0], 'ro')  # Note that matplotlib uses x, y coordinates

        # Draw rectangle around patch
        rect1 = patches.Rectangle((coord1[1] - window_size // 2, coord1[0] - window_size // 2),
                                  window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect1)

        # Second image
        axes[1].imshow(image2, cmap='gray', extent=[0, 64, 64, 0])
        axes[1].set_title(title)
        axes[1].axis('off')
        axes[1].plot(coord2[1], coord2[0], 'ro')

        rect2 = patches.Rectangle((coord2[1] - window_size // 2, coord2[0] - window_size // 2),
                                  window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect2)

        plt.show()

        # Display the patches and SSIM value
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
        axes2[0].imshow(patch1, cmap='gray', extent=[0, 64, 64, 0])
        axes2[0].set_title('Patch from Image 1')
        axes2[0].axis('off')
        axes2[1].imshow(patch2, cmap='gray', extent=[0, 64, 64, 0])
        axes2[1].set_title('Patch from Image 2')
        axes2[1].axis('off')
        plt.suptitle(f'SSIM between patches: {ssim_value:.4f}', fontsize=16)
        plt.show()

    @staticmethod
    def are_pixels_similar(image1, image2, coord1, coord2, window_size=7, threshold=0.5):
        """
        Determines if two pixels in different images are similar based on the SSIM of patches around them.

        Parameters:
        - mes1, mes2: Two set of measurements that has been done.
        - coord1, coord2: Tuples (y, x) representing the coordinates of the pixels in image1 and image2.
        - window_size: Size of the square window around the pixel to compute SSIM.
        - threshold: SSIM threshold above which pixels are considered similar.

        Returns:
        - ssim_value: The computed SSIM value.
        - is_similar: True if SSIM between patches is above the threshold, False otherwise.
        - patch1, patch2: The patches extracted from image1 and image2.
        """
        y1, x1 = coord1
        y2, x2 = coord2
        half_window = window_size // 2

        # image1 = self.data_dict[mes1].get_live_scan()
        # image2 = self.data_dict[mes2].get_live_scan()

        patch1 = image1[max(0, y1 - half_window): y1 + half_window + 1,
                 max(0, x1 - half_window): x1 + half_window + 1]
        patch2 = image2[max(0, y2 - half_window): y2 + half_window + 1,
                 max(0, x2 - half_window): x2 + half_window + 1]

        min_rows = min(patch1.shape[0], patch2.shape[0])
        min_cols = min(patch1.shape[1], patch2.shape[1])
        patch1 = patch1[:min_rows, :min_cols]
        patch2 = patch2[:min_rows, :min_cols]

        ssim_value = ssim(patch1, patch2,
                          data_range=patch1.max() - patch1.min(),
                          channel_axis=-1 if patch1.ndim == 3 else None)

        is_similar = ssim_value >= threshold
        # CondAns.visualize_pixel_similarity(image1, image2, coord1, coord2, patch1, patch2, ssim_value,
        # window_size=window_size)
        return ssim_value, is_similar, patch1, patch2

    @staticmethod
    def map_pixels(img1, img2, window_size=11, search_radius=15):
        """
        Find pixel correspondences between two images using local SSIM comparison

        Args:
            img1 (numpy.ndarray): First image (2D array)
            img2 (numpy.ndarray): Second image (2D array)
            window_size (int): Odd number size of the comparison window
            search_radius (int): Search radius in pixels around original position

        Returns:
            correspondence_map (numpy.ndarray): Array of shape (H, W, 2) containing
            corresponding [x,y] coordinates in img2 for each pixel in img1
        """

        assert img1.ndim == 2 and img2.ndim == 2, "Images must be 2D arrays"
        assert img1.shape == img2.shape, "Images must have the same dimensions"
        assert window_size % 2 == 1, "Window size must be odd"

        global_min = min(img1.min(), img2.min())
        global_max = max(img1.max(), img2.max())
        data_range = global_max - global_min
        result_dict = dict()

        pad = window_size // 2
        height, width = img1.shape

        img1_padded = np.pad(img1, pad, mode='reflect')
        img2_padded = np.pad(img2, pad, mode='reflect')

        correspondence_map = np.zeros((height, width, 2), dtype=np.int32)

        offsets = [(di, dj) for di in range(-search_radius, search_radius + 1)
                   for dj in range(-search_radius, search_radius + 1)]

        for i in range(height):
            for j in range(width):
                pi, pj = i + pad, j + pad

                window1 = img1_padded[pi - pad:pi + pad + 1, pj - pad:pj + pad + 1]

                best_score = -np.inf
                best_pos = (i, j)

                min_i = max(pad, pi - search_radius)
                max_i = min(img2_padded.shape[0] - pad, pi + search_radius + 1)
                min_j = max(pad, pj - search_radius)
                max_j = min(img2_padded.shape[1] - pad, pj + search_radius + 1)

                for x in range(min_i, max_i):
                    for y in range(min_j, max_j):
                        window2 = img2_padded[x - pad:x + pad + 1, y - pad:y + pad + 1]

                        score = ssim(window1, window2,
                                     data_range=data_range,
                                     gaussian_weights=True,
                                     win_size=window_size,
                                     use_sample_covariance=False)

                        if score > best_score:
                            best_score = score
                            best_pos = (x - pad, y - pad)

                correspondence_map[i, j] = best_pos

        for i in range(height):
            for j in range(width):
                result_dict[(i, j)] = tuple(correspondence_map[i][j])

        return result_dict

    @staticmethod
    def fit_lorentzian_spectrum(x, y, num_peaks=1, model_func=VoigtModel):
        """
        Fits the given spectrum using a sum of Lorentzian functions.

        Parameters:
            x (numpy.ndarray): The x-axis data (e.g., wavelength, energy, etc.).
            y (numpy.ndarray): The y-axis data (intensity).
            num_peaks (int): Number of Lorentzian peaks to fit.
            model_func (class): The model class to use for each peak (e.g., VoigtModel, GaussianModel).

        Returns:
            lmfit.model.ModelResult: The fitting result.
        """
        composite_model = None
        params = None

        for i in range(num_peaks):
            model = model_func(prefix=f'p{i}_')

            if composite_model is None:
                composite_model = model
            else:
                composite_model += model

            if i == 0:
                amp_guess = 2000
                center_guess = 800
                sigma_guess = 10
            elif i == 1:
                amp_guess = 2000
                center_guess = 900
                sigma_guess = 1
            elif i == 2:
                amp_guess = 2000
                center_guess = 970
                sigma_guess = 1

            model_params = model.make_params()
            if params is None:
                params = model_params
            else:
                params.update(model_params)

            params[f'p{i}_amplitude'].set(value=amp_guess, min=0)
            params[f'p{i}_center'].set(value=center_guess, min=x.min(), max=x.max())
            params[f'p{i}_sigma'].set(value=sigma_guess, min=0)

        result = composite_model.fit(y, params, x=x)
        return result

    @staticmethod
    def __plot_image_with_rect(ax, image_data, coord, title):
        """
        Plot a grayscale image on the given axis with a red rectangle
        marking the coordinate.

        Parameters:
            ax (matplotlib.axes.Axes): Axis to plot the image.
            image_data (numpy.ndarray): 2D image array.
            coord (tuple): (row, col) coordinate. If None, uses (0, 0).
            title (str): Axis title.
        """
        if coord == (0, 0):
            ax.imshow(image_data, extent=(0, image_data.shape[1], image_data.shape[0], 0),
                      cmap='gray')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        else:
            ax.imshow(image_data, extent=(0, image_data.shape[1], image_data.shape[0], 0),
                      cmap='gray')
            rect = patches.Rectangle((coord[1], coord[0]), 1, 1,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        return coord, ax
