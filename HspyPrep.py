# Implemented by M.Hassanzadeh

# This library is implemented to preprocess the hyperspectral data
# Also with this library we can extract and get the initial visualization of the data


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import matplotlib.patches as patches
from scipy.signal import medfilt
from PIL import Image
import hyperspy.api as hs
import lumispy as lsp
from scipy.optimize import curve_fit
import os
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
from sklearn.mixture import GaussianMixture
from matplotlib.ticker import ScalarFormatter
from traits.trait_types import self
from matplotlib.ticker import LogFormatter
from matplotlib.colors import LogNorm

class HspyPrep:
    
    def __init__(self, file_path, step, whole_seconds, contain_bg=False):
        """
        Constructor of the class
        give the filepath exactly in this format: 'Data/Path/To/The/File/'
        :param file_path: (string) path of the file that contains the HYPCard.sur file
        """
        self.live_scan = None
        self.optimal_ty = None
        self.optimal_tx = None
        self.optimal_theta = None
        self.step = step
        self.whole_seconds = whole_seconds
        self.file_path = file_path
        self.hsp_obj_file_path = hs.load(self.file_path + 'HYPCard.sur')
        # --------CL_Correction--------
        data = self.hsp_obj_file_path.data
        rearranged_data = np.roll(data, 1, axis=1)
        self.hsp_obj_file_path.data = rearranged_data
        # --------CL_Correction--------
        live_scan_file_name = [f for f in os.listdir(self.file_path) if f.startswith("Live") and f.endswith('.sur')]
        # self.live_scan = np.array(Image.open(self.file_path + live_scan_file_name[0]))
        # self.live_scan = self.manual_hist_equalization(self.live_scan)
        # self.live_scan = self.live_scan * -1
        self.live_scan = self.hyperspy_to_numpy(hs.load(self.file_path + live_scan_file_name[0]))
        self.live_scan = np.roll(self.live_scan, 1, axis=1)

        self.dataframe_obj = self.hyperspy_to_numpy(self.hsp_obj_file_path)
        self.se_after = self.hyperspy_to_numpy(hs.load(self.file_path + 'SE_After-SE.sur'))
        self.se_before = self.hyperspy_to_numpy(hs.load(self.file_path + 'SE_Before-SE.sur'))
        se_file_name = [f for f in os.listdir(file_path) if f.startswith("SE") and f.endswith(".JPG")]
        self.se_image_file_path = self.file_path + se_file_name[0]
        if contain_bg:
            bg_file_name = [f for f in os.listdir(os.path.dirname(file_path[:-1])) if
                            f.startswith("BG") and f.endswith(".txt")]
            with open('/'.join(file_path.split('/')[:-2]) + '/' + bg_file_name[0], 'r') as f:
                l = []
                for line in f:
                    l.append(line.split())
                self.background = np.array(l, dtype=float)[:, ::-1]
                f.close()

    def plot_se_image(self, title):
        """
        This function plots the SE image
        :return: None
        """

        plt.imshow(plt.imread(self.se_image_file_path), cmap='gray')
        plt.grid(False)
        plt.title(title)
        plt.show()

    def wavelength_to_energy(self, wavelength_nm):
        h = 4.135667696e-15
        c = 3e8

        wavelength_m = wavelength_nm * 1e-9
        energy_eV = (h * c) / wavelength_m
        return energy_eV

    def plot_spectrum(self, x, y, title='Spectrum', data=None, label_data=None, save_fig=False, address=None):
        # assert self.optimal_tx is not None, "First you need to call the transition matrix"

        rect_size_left = 1

        fig, axd = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']],
                                      constrained_layout=True, figsize=[15, 15])

        plt.rcParams.update(
            {'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

        axd['left'].imshow(self.live_scan, extent=(0, self.live_scan.shape[1], self.live_scan.shape[0], 0), cmap='gray')
        axd['left'].set_title('Live Scan of the SEM')
        rect = patches.Rectangle((x, y), rect_size_left, rect_size_left, linewidth=0.8, edgecolor='red', facecolor='none')
        axd['left'].add_patch(rect)
        axd['left'].set_xlabel("X-axis (pixels)")
        axd['left'].set_ylabel("Y-axis (pixels)")
        axd['left'].grid(False)

        axd['bottom'].plot(self.hsp_obj_file_path.axes_manager[2].axis, self.dataframe_obj[y][x],
                           label='Original Spectrum', color='red', linewidth=2)
        if data is not None and label_data != 'Peaks':
            axd['bottom'].plot(self.hsp_obj_file_path.axes_manager[2].axis, data, label=label_data, color='blue',
                               linestyle='--', linewidth=2)
        elif label_data == 'Peaks':
            axd['bottom'].plot(self.hsp_obj_file_path.axes_manager[2].axis[data], self.dataframe_obj[y][x][data], 'bx',
                               label='Detected Peaks', markersize=8)

        # axd['bottom'].set_title(title)
        axd['bottom'].set_xlabel('Wavelength (nm)')
        axd['bottom'].set_ylabel('Intensity (a.u.)')
        axd['bottom'].legend()
        axd['bottom'].grid(True, linestyle='--', linewidth=0.7)

        wavelength_axis = self.hsp_obj_file_path.axes_manager[2].axis
        ax_bottom = axd['bottom']
        ax_top = ax_bottom.secondary_xaxis('top')
        ax_top.set_xlabel('Energy (eV)')
        energy_ticks = np.linspace(wavelength_axis[0], wavelength_axis[-1], num=5)
        ax_top.set_xticks(energy_ticks)
        ax_top.set_xticklabels([f'{1239.84193 / wl:.2f}' for wl in energy_ticks])

        if save_fig:
            plt.savefig(address, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_heat_map_emission_spectra(self, x, y, peak_ranges: tuple, title='Spectrum', height=50, prominence=30,
                                       vmin=None, vmax=None):

        assert self.optimal_tx is not None, "First you need to call the transition matrix"
        output = np.zeros((self.dataframe_obj.shape[0], self.dataframe_obj.shape[0], 1))

        for i in range(self.dataframe_obj.shape[0]):
            for j in range(self.dataframe_obj.shape[0]):
                spectrum = self.dataframe_obj[i, j, :]
                output[i, j, 0] = HspyPrep.process_spectrum(spectrum, self.get_wavelengths(), peak_ranges,
                                                            height=height, prominence=prominence)

        rect_size = 1

        fig, axd = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']],
                                      constrained_layout=True, figsize=[13, 13])

        axd['left'].imshow(self.live_scan, extent=(0, self.live_scan.shape[1], self.live_scan.shape[0], 0),
                           cmap='gray')
        axd['left'].set_title('Live Scan of the SEM')

        top_left = (x, y)
        rect = patches.Rectangle(top_left, rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
        axd['left'].add_patch(rect)
        axd['left'].set_title("Image with Rectangle")
        axd['left'].set_xlabel("X-axis (pixels)")
        axd['left'].set_ylabel("Y-axis (pixels)")
        img2 = axd['left'].imshow(output[:, :, 0], cmap='inferno', alpha=0.6, vmin=vmin, vmax=vmax,
                                  extent=(0, self.live_scan.shape[1], self.live_scan.shape[0], 0))
        cbar2 = fig.colorbar(img2, ax=axd['left'], orientation='vertical', fraction=0.045, pad=0.0001)
        cbar2.formatter = ScalarFormatter(useMathText=True)
        cbar2.formatter.set_scientific(True)
        cbar2.formatter.set_powerlimits((-1, 1))
        cbar2.update_ticks()

        axd['left'].grid(False)

        # -------------------Correction-------------------
        index_calc = (self.live_scan.shape[0] * y) + x
        transition_array = self.create_transition_with_matrix(self.se_before, self.se_after, self.step,
                                                              self.whole_seconds,
                                                              self.optimal_tx, self.optimal_ty,
                                                              self.optimal_theta, pool_size=1, frame_number=index_calc)
        # -------------------Correction-------------------

        # shrinking_factor = int(transition_array.shape[0]) / int(self.live_scan.shape[0])
        pixel_size = transition_array.shape[0] / self.live_scan.shape[0]
        rect_size_right = pixel_size
        start_x = (x * pixel_size)
        start_y = (y * pixel_size)

        axd['right'].imshow(transition_array[:, :, index_calc], cmap='gray',
                            extent=(0, transition_array.shape[1], transition_array.shape[0], 0))
        axd['right'].set_title('Transitioned Image (Exact Location of Measurement)')

        top_left = (start_x, start_y)
        rect = patches.Rectangle(top_left, rect_size_right, rect_size_right, linewidth=1, edgecolor='r',
                                 facecolor='none')
        axd['right'].add_patch(rect)
        axd['right'].set_xlabel("X-axis (pixels)")
        axd['right'].set_ylabel("Y-axis (pixels)")
        axd['right'].grid(False)

        axd['bottom'].plot(self.hsp_obj_file_path.axes_manager[2].axis, self.dataframe_obj[y][x],
                           label='Original Spectrum',
                           color='red')

        axd['bottom'].set_title(title)
        axd['bottom'].set_xlabel('Wavelength (nm)')
        axd['bottom'].set_ylabel('Intensity (a.u.)')
        axd['bottom'].legend()

        plt.show()

    def plot_heat_map_emission_spectra_range(self, x, y, peak_ranges: tuple, title='Spectrum', vmin=None, vmax=None):

        assert self.optimal_tx is not None, "First you need to call the transition matrix"
        output = np.zeros((self.dataframe_obj.shape[0], self.dataframe_obj.shape[0], 1))

        for i in range(self.dataframe_obj.shape[0]):
            for j in range(self.dataframe_obj.shape[0]):
                spectrum = self.dataframe_obj[i, j, :]
                output[i, j, 0] = HspyPrep.sum_counts_in_range(spectrum, self.get_wavelengths(), peak_ranges)

        rect_size = min(self.live_scan.shape[0], self.live_scan.shape[1]) // self.live_scan.shape[0]

        fig, axd = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']],
                                      constrained_layout=True, figsize=[13, 13])
        axd['left'].imshow(self.live_scan, extent=(0, self.live_scan.shape[1], self.live_scan.shape[0], 0),
                           cmap='gray')
        axd['left'].set_title('Live Scan of the SEM')

        top_left = (x - rect_size // 2, y - rect_size // 2)
        rect = patches.Rectangle(top_left, rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
        axd['left'].add_patch(rect)
        axd['left'].set_title("Image with Rectangle")
        axd['left'].set_xlabel("X-axis (pixels)")
        axd['left'].set_ylabel("Y-axis (pixels)")
        img2 = axd['left'].imshow(output[:, :, 0], cmap='inferno', alpha=0.6, vmin=vmin, vmax=vmax,
                                  extent=(0, self.live_scan.shape[1], self.live_scan.shape[0], 0))
        cbar2 = fig.colorbar(img2, ax=axd['left'], orientation='vertical', fraction=0.045, pad=0.0001)
        cbar2.formatter = ScalarFormatter(useMathText=True)
        cbar2.formatter.set_scientific(True)
        cbar2.formatter.set_powerlimits((-1, 1))
        cbar2.update_ticks()

        axd['left'].grid(False)

        # -------------------Correction-------------------
        index_calc = (self.live_scan.shape[0] * y) + x
        transition_array = self.create_transition_with_matrix(self.se_before, self.se_after, self.step,
                                                              self.whole_seconds,
                                                              self.optimal_tx, self.optimal_ty,
                                                              self.optimal_theta, pool_size=1, frame_number=index_calc)
        # -------------------Correction-------------------
        shrinking_factor = int(transition_array.shape[0]) / int(self.live_scan.shape[0])

        rect_size = rect_size * shrinking_factor
        start_x = (x * shrinking_factor) + (shrinking_factor / 2)
        start_y = (y * shrinking_factor) + (shrinking_factor / 2)

        axd['right'].imshow(transition_array[:, :, index_calc], cmap='gray')
        axd['right'].set_title('Transitioned Image (Exact Location of Measurement')

        top_left = (start_x - rect_size // 2, start_y - rect_size // 2)
        rect = patches.Rectangle(top_left, rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
        axd['right'].add_patch(rect)
        axd['right'].set_xlabel("X-axis (pixels)")
        axd['right'].set_ylabel("Y-axis (pixels)")
        axd['right'].grid(False)

        axd['bottom'].plot(self.hsp_obj_file_path.axes_manager[2].axis, self.dataframe_obj[y][x],
                           label='Original Spectrum',
                           color='red')

        axd['bottom'].set_title(title)
        axd['bottom'].set_xlabel('Wavelength (nm)')
        axd['bottom'].set_ylabel('Intensity (a.u.)')
        axd['bottom'].legend()

        plt.show()

    def plot_heatmap(self, ranges: tuple, vmin=None, vmax=None, title="Heatmap of Summed Intensities"):

        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
        })

        output = np.zeros((self.dataframe_obj.shape[0], self.dataframe_obj.shape[1]))
        for i in range(self.dataframe_obj.shape[0]):
            for j in range(self.dataframe_obj.shape[1]):
                spectrum = self.dataframe_obj[i, j, :]
                output[i, j] = HspyPrep.sum_counts_in_range(
                    spectrum, self.get_wavelengths(), ranges
                )

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(output,
                    cmap='inferno',
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    origin='upper')

        ax.set_title(title)
        ax.set_xlabel("X-axis (pixels)")
        ax.set_ylabel("Y-axis (pixels)")
        ax.grid(False)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(LogFormatter())
        cbar.update_ticks()

        plt.show()



    def get_transitioned_image(self, x, y):
        index_calc = (self.live_scan.shape[0] * y) + x
        transition_array = self.create_transition_with_matrix(self.se_before, self.se_after, self.step,
                                                              self.whole_seconds,
                                                              self.optimal_tx, self.optimal_ty,
                                                              self.optimal_theta, pool_size=1, frame_number=index_calc)
        return transition_array

    def calculate_index(self, x, y):
        index_calc = (self.live_scan.shape[0] * y) + x
        loss = np.inf
        optimized_index = -1
        pool_size = int(self.se_after.shape[0] / self.live_scan.shape[0])
        for i in range(index_calc - 640, index_calc + 641, 32):
            transition_array = self.create_transition_with_matrix(self.se_before, self.se_after, self.step,
                                                                  self.whole_seconds,
                                                                  self.optimal_tx, self.optimal_ty,
                                                                  self.optimal_theta, pool_size=1, frame_number=i)

            averaged_transition_array = HspyPrep.average_pooling(transition_array[:, :, i], (pool_size, pool_size))

            diff = (self.live_scan[y - 1:y + 2, x - 1:x + 2] - averaged_transition_array[y - 2:y + 3, x - 2:x + 3]) ** 2
            if np.sum(diff) < loss:
                optimized_index = i
        return optimized_index

    def apply_filter_noises(self, kernel_size=7):
        """
        This function applies the filter to the data to remove the noises and spikes
        :return: (numpy array) filtered data
        """
        for i in range(self.dataframe_obj.shape[0]):
            for j in range(self.dataframe_obj.shape[1]):
                self.dataframe_obj[i][j] = medfilt(self.dataframe_obj[i][j], kernel_size=kernel_size)
                self.hsp_obj_file_path.data[i][j] = self.dataframe_obj[i][j]


    def remove_background(self, file_path=None):
        """
        In case you want to give the file path of the background.
        If you have entered the file path during initialization, this function will remove the background automatically.
        Subtracts background from each spectrum in spectra.
        """
        if file_path is not None:
            with open(file_path, 'r') as f:
                l = []
                for line in f:
                    l.append(line.split())
                self.background = np.array(l, dtype=float)[:, ::-1]
                f.close()
        for i in range(self.dataframe_obj.shape[0]):
            for j in range(self.dataframe_obj.shape[1]):
                self.dataframe_obj[i][j] = self.dataframe_obj[i][j] - self.background[1]
                self.hsp_obj_file_path.data[i][j] = self.dataframe_obj[i][j]

        # self.plot_spectrum(0, 0, 'Removed Background')

    def get_numpy_spectra(self):
        """
        This function returns the numpy array of the hyperspectral data
        :return: (numpy array) numpy array of the hyperspectral data
        """
        return self.dataframe_obj

    def get_wavelengths(self):
        """

        :return:
        """

        return self.hsp_obj_file_path.axes_manager[2].axis

    def get_live_scan(self):
        """
        Get a live scan
        :return: numpy array of live scan
        """
        return self.live_scan

    def get_peak(self, x, y, height, prominence):
        """
        Get a peak point in nm
        :param x:
        :param y:
        :param height:
        :param prominence:
        :return:
        """
        peaks, _ = find_peaks(self.dataframe_obj[y][x], height=height, prominence=prominence)
        return peaks

    def get_se_before_image(self):
        return self.se_before

    def get_se_after_image(self):
        return self.se_after

    def get_hyperspy_obj(self):
        return self.hsp_obj_file_path

    @staticmethod
    def hyperspy_to_numpy(hsp_obj: lsp.signals.cl_spectrum.CLSpectrum):
        """
        This function converts the hyperspy object to numpy array
        :return: (numpy array) numpy array of the hyperspy object
        """
        return np.array(hsp_obj.data)

    @staticmethod
    def apply_transformation(image, theta, tx, ty):
        theta_rad = np.radians(theta)
        transformation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), tx],
            [np.sin(theta_rad), np.cos(theta_rad), ty]
        ])
        transformed_image = affine_transform(image, transformation_matrix[:2, :2], offset=[tx, ty], mode='constant',
                                             cval=0)

        return transformed_image

    @staticmethod
    def max_pooling(image, pool_size):
        """
        Applies max pooling to reduce the size of the input image.

        Args:
            image (numpy array): The input 2D image (512, 512).
            pool_size (tuple): The size of the pooling window (height, width).

        Returns:
            numpy array: The pooled image.
        """
        h, w = image.shape
        pool_h, pool_w = pool_size
        new_h = h // pool_h
        new_w = w // pool_w
        pooled_image = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                window = image[i * pool_h:(i + 1) * pool_h, j * pool_w:(j + 1) * pool_w]
                pooled_image[i, j] = np.max(window)
        return pooled_image

    @staticmethod
    def average_pooling(image, pool_size):
        """
        Applies average pooling to reduce the size of the input image.

        Args:
            image (numpy array): The input 2D image (512, 512).
            pool_size (tuple): The size of the pooling window (height, width).

        Returns:
            numpy array: The pooled image.
        """
        h, w = image.shape
        pool_h, pool_w = pool_size
        new_h = h // pool_h
        new_w = w // pool_w
        pooled_image = np.zeros((new_h, new_w))
        for i in range(new_h):
            for j in range(new_w):
                window = image[i * pool_h:(i + 1) * pool_h, j * pool_w:(j + 1) * pool_w]
                pooled_image[i, j] = np.mean(window)
        return pooled_image

    @staticmethod
    def loss_function(params, image1, image2):
        theta, tx, ty = params
        transformed_image = HspyPrep.apply_transformation(image2, theta, tx, ty)

        diff = (image1 - transformed_image) ** 2
        return np.sum(diff)

    @staticmethod
    def pure_loss_function(image1, image2):
        diff = (image1 - image2) ** 2
        return np.sum(diff)

    @staticmethod
    def process_spectrum(spectrum, wavelengths, peak_range, height, prominence):
        indices = np.where((wavelengths >= peak_range[0]) & (wavelengths <= peak_range[1]))[0]
        peaks, _ = find_peaks(spectrum[indices], height=height, prominence=prominence)

        if len(peaks) == 0:
            return 0
        avg_peak_idx = indices[peaks].mean().astype(int)

        upper_bound = np.argmin(np.abs(wavelengths - (wavelengths[avg_peak_idx] - 10)))
        lower_bound = np.argmin(np.abs(wavelengths - (wavelengths[avg_peak_idx] + 10)))

        return spectrum[lower_bound:upper_bound + 1].sum()

    @staticmethod
    def sum_counts_in_range(spectrum, wavelengths, peak_range):
        indices = np.where((wavelengths >= peak_range[0]) & (wavelengths <= peak_range[1]))[0]
        return spectrum[indices].sum()

    @staticmethod
    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def create_transition_with_matrix(image1, image2, step, whole_seconds, opt_tx, opt_ty, opt_theta, pool_size,
                                      frame_number):
        """
        Creates a transition between two images using a transformation matrix.
        """
        # assert image1.shape == image2.shape, "Both images must have the same shape"

        num_frames = int(step * whole_seconds)
        transition_array = np.zeros((image1.shape[0], image1.shape[1], num_frames))

        theta = opt_theta / num_frames
        tx = opt_tx / num_frames
        ty = opt_ty / num_frames

        transformed_image_after = HspyPrep.apply_transformation(image1, frame_number * theta,
                                                                frame_number * tx / pool_size,
                                                                frame_number * ty / pool_size)

        transformed_image_before = HspyPrep.apply_transformation(image2, (frame_number - num_frames) * theta,
                                                                 (frame_number - num_frames) * tx / pool_size,
                                                                 (frame_number - num_frames) * ty / pool_size)

        mask = (transformed_image_after == 0)
        filled_image = transformed_image_after.copy()
        filled_image[mask] = transformed_image_before[mask]

        transition_array[:, :, frame_number] = filled_image

        return transition_array

    @staticmethod
    def manual_hist_equalization(image):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        cdf = hist.cumsum()

        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())

        cdf = np.ma.filled(cdf_masked, 0).astype('uint8')

        equalized_image = cdf[image]

        return equalized_image
