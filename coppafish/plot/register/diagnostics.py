import os
import nd2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import napari
from qtpy.QtCore import Qt
from superqt import QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLineEdit, QLabel
from PyQt5.QtGui import QFont
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ...setup import Notebook
from skimage.filters import window, sobel
from coppafish.register import preprocessing
from coppafish.register.base import huber_regression, brightness_scale, ols_regression
from coppafish.utils import tiles_io
from scipy.ndimage import affine_transform
from scipy import fft
import itertools


plt.style.use("dark_background")

# there are 2 main parts to the reg pipeline:
# 1. optical flow
# 2. icp correction


class RegistrationViewer:
    def __init__(self, nb: Notebook, t: int = None):
        """
        Function to overlay tile, round and channel with the anchor in napari and view the registration.
        This function is only long because we have to convert images to zyx and the transform to zyx * zyx
        Args:
            nb: Notebook
            t: common tile
        """
        # initialise frequently used variables, attaching those which are otherwise awkward to recalculate to self
        nbp_file, nbp_basic = nb.file_names, nb.basic_info
        self.nb = nb
        use_rounds, use_channels = (nbp_basic.use_rounds + nbp_basic.use_preseq * [nbp_basic.pre_seq_round],
                                    nbp_basic.use_channels)
        self.r_ref, self.c_ref = nbp_basic.anchor_round, nb.basic_info.anchor_channel
        self.round_registration_channel = nbp_basic.dapi_channel
        self.r_mid = len(use_rounds) // 2
        y_mid, x_mid = nbp_basic.tile_centre[:-1]
        z_mid = len(nbp_basic.use_z) // 2
        self.new_origin = np.array([y_mid - 250, x_mid - 250, z_mid - 5])
        # Initialise file directories
        self.target_round_image, self.target_channel_image = [], []
        self.base_round_image, self.base_channel_image = None, None
        self.output_dir = os.path.join(nbp_file.output_dir, 'reg_images/')
        self.viewer = napari.Viewer()

        # Next lines will be cleaning up the napari viewer and adding the sliders and buttons
        # Make layer list invisible to remove clutter
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        # Now we will create 2 sliders. One will control all the contrast limits simultaneously, the other all anchor
        # images simultaneously.
        self.im_contrast_limits_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.anchor_contrast_limits_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.im_contrast_limits_slider.setRange(0, 256)
        self.anchor_contrast_limits_slider.setRange(0, 256)
        # Set default lower limit to 0 and upper limit to 255
        self.im_contrast_limits_slider.setValue((0, 255))
        self.anchor_contrast_limits_slider.setValue((0, 255))

        # Now we run a method that sets these contrast limits using napari
        self.viewer.window.add_dock_widget(self.im_contrast_limits_slider, area="left", name='Imaging Contrast')
        self.viewer.window.add_dock_widget(self.anchor_contrast_limits_slider, area="left", name='Anchor Contrast')
        # Now create events that will recognise when someone has changed slider values
        self.anchor_contrast_limits_slider.valueChanged.connect(lambda x: self.change_anchor_layer_contrast(x[0], x[1]))
        self.im_contrast_limits_slider.valueChanged.connect(lambda x: self.change_imaging_layer_contrast(x[0], x[1]))

        # Add a single button to turn off the base images and a single button to turn off the target images
        self.switch_buttons = ButtonOnOffWindow()
        # This allows us to link clickng to slot functions
        self.switch_buttons.button_base.clicked.connect(self.button_base_images_clicked)
        self.switch_buttons.button_target.clicked.connect(self.button_target_images_clicked)
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.switch_buttons, area="left", name="Switch", add_vertical_stretch=False)

        # Add buttons to change between registration methods
        self.method_buttons = ButtonMethodWindow("SVR")
        # This allows us to link clickng to slot functions
        self.method_buttons.button_icp.clicked.connect(self.button_icp_clicked)
        self.method_buttons.button_svr.clicked.connect(self.button_svr_clicked)
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.method_buttons, area="left", name="Method")

        # Add buttons to select different tiles. Involves initialising variables use_tiles and tilepos
        tilepos_xy = np.roll(self.nb.basic_info.tilepos_yx, shift=1, axis=1)
        # Invert y as y goes downwards in the set geometry func
        num_rows = np.max(tilepos_xy[:, 1])
        tilepos_xy[:, 1] = num_rows - tilepos_xy[:, 1]
        # get use tiles
        use_tiles = self.nb.basic_info.use_tiles
        # If no tile provided then default to the first tile in use
        if t is None:
            t = use_tiles[0]
        # Store a copy of the working tile in the RegistrationViewer
        self.tile = t

        # Now create tile_buttons
        self.tile_buttons = ButtonTileWindow(tile_pos_xy=tilepos_xy, use_tiles=use_tiles, active_button=self.tile)
        for tile in use_tiles:
            # Now connect the button associated with tile t to a function that activates t and deactivates all else
            self.tile_buttons.__getattribute__(str(tile)).clicked.connect(self.create_tile_slot(tile))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.tile_buttons, area="left", name="Tiles", add_vertical_stretch=False)

        # We want to create a single napari widget containing buttons which for each round and channel

        # Create all buttons for round registration
        self.flow_buttons = ButtonFlowWindow(use_rounds)
        for rnd in use_rounds:
            # now connect this to a slot that will activate the round flow viewer
            self.flow_buttons.__getattribute__(f'R{rnd}').clicked.connect(self.create_round_slot(rnd))
        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.flow_buttons, area="left", name='SVR Diagnostics',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for ICP diagnostics
        self.icp_buttons = ButtonICPWindow()
        # Now connect buttons to functions
        self.icp_buttons.button_mse.clicked.connect(self.button_mse_clicked)
        self.icp_buttons.button_matches.clicked.connect(self.button_matches_clicked)
        self.icp_buttons.button_deviations.clicked.connect(self.button_deviations_clicked)

        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(
            self.icp_buttons, area="left", name="ICP Diagnostics", add_vertical_stretch=False
        )

        # Create a single widget containing buttons for Overlay diagnostics
        self.overlay_buttons = ButtonOverlayWindow()
        # Now connect button to function
        self.overlay_buttons.button_overlay.clicked.connect(self.view_button_clicked)
        # Add buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(
            self.overlay_buttons, area="left", name="Overlay Diagnostics", add_vertical_stretch=False
        )

        # Create a single widget containing buttons for BG Subtraction diagnostics if bg subtraction has been run
        if self.nb.basic_info.use_preseq:
            self.bg_sub_buttons = ButtonBGWindow()
            # Now connect buttons to function
            self.bg_sub_buttons.button_overlay.clicked.connect(self.button_bg_sub_overlay_clicked)
            self.bg_sub_buttons.button_brightness_scale.clicked.connect(self.button_brightness_scale_clicked)
            # Add buttons as widgets in napari viewer
            self.viewer.window.add_dock_widget(
                self.bg_sub_buttons, area="left", name="BG Subtraction Diagnostics", add_vertical_stretch=False
            )

        # Create a widget containing buttons for fluorescent bead diagnostics if fluorescent beads have been used
        if self.nb.file_names.fluorescent_bead_path is not None:
            self.bead_buttons = ButtonBeadWindow()
            # Now connect buttons to function
            self.bead_buttons.button_fluorescent_beads.clicked.connect(self.button_fluorescent_beads_clicked)
            # Add buttons as widgets in napari viewer
            self.viewer.window.add_dock_widget(
                self.bead_buttons, area="left", name="Fluorescent Bead Diagnostics", add_vertical_stretch=False
            )

        # Get target images and anchor image
        self.get_images()

        # Plot images
        self.plot_images()

        napari.run()

    # Button functions

    # Tiles grid
    def create_tile_slot(self, t):

        def tile_button_clicked():
            # We're going to connect each button str(t) to a function that sets checked str(t) and nothing else
            # Also sets self.tile = t
            use_tiles = self.nb.basic_info.use_tiles
            for tile in use_tiles:
                self.tile_buttons.__getattribute__(str(tile)).setChecked(tile == t)
            self.tile = t
            self.update_plot()

        return tile_button_clicked

    def button_base_images_clicked(self):
        n_images = len(self.viewer.layers)
        for i in range(0, n_images, 2):
            self.viewer.layers[i].visible = self.switch_buttons.button_base.isChecked()

    def button_target_images_clicked(self):
        n_images = len(self.viewer.layers)
        for i in range(1, n_images, 2):
            self.viewer.layers[i].visible = self.switch_buttons.button_target.isChecked()

    # Contrast
    def change_anchor_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(0, len(self.viewer.layers), 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    # contrast
    def change_imaging_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(1, len(self.viewer.layers), 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    # method
    def button_svr_clicked(self):
        # Only allow one button pressed
        # Below does nothing if method is already svr and updates plot otherwise
        if self.method_buttons.method == "SVR":
            self.method_buttons.button_svr.setChecked(True)
            self.method_buttons.button_icp.setChecked(False)
        else:
            self.method_buttons.button_svr.setChecked(True)
            self.method_buttons.button_icp.setChecked(False)
            self.method_buttons.method = "SVR"
            # Because method has changed, also need to change transforms
            # Update set of transforms
            self.transform = self.nb.register.initial_transform
            self.update_plot()

    # method
    def button_icp_clicked(self):
        # Only allow one button pressed
        # Below does nothing if method is already icp and updates plot otherwise
        if self.method_buttons.method == "ICP":
            self.method_buttons.button_icp.setChecked(True)
            self.method_buttons.button_svr.setChecked(False)
        else:
            self.method_buttons.button_icp.setChecked(True)
            self.method_buttons.button_svr.setChecked(False)
            self.method_buttons.method = "ICP"
            # Because method has changed, also need to change transforms
            # Update set of transforms
            self.transform = self.nb.register.transform
            self.update_plot()

    # Flow
    def create_round_slot(self, r):

        def round_button_clicked():
            use_rounds = self.nb.basic_info.use_rounds + self.nb.basic_info.use_preseq * [self.nb.basic_info.pre_seq_round]
            for rnd in use_rounds:
                self.flow_buttons.__getattribute__('R' + str(rnd)).setChecked(rnd == r)
            # We don't need to update the plot, we just need to call the viewing function
            view_round_regression_scatter(nb=self.nb, t=self.tile, r=r)

        return round_button_clicked

    # outlier removal
    def button_vec_field_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_vec_field_r.setChecked(True)
        shift_vector_field(nb=self.nb, round=True)

    # outlier removal
    def button_vec_field_c_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_vec_field_c.setChecked(True)
        shift_vector_field(nb=self.nb, round=False)

    # outlier removal
    def button_shift_cmap_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_shift_cmap_r.setChecked(True)
        zyx_shift_image(nb=self.nb, round=True)

    # outlier removal
    def button_shift_cmap_c_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_shift_cmap_c.setChecked(True)
        zyx_shift_image(nb=self.nb, round=False)

    # outlier removal
    def button_scale_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_scale_r.setChecked(True)
        view_round_scales(nb=self.nb)

    # outlier removal
    def button_scale_c_clicked(self):
        view_channel_scales(nb=self.nb)

    # icp
    def button_mse_clicked(self):
        view_icp_mse(nb=self.nb, t=self.tile)

    # icp
    def button_matches_clicked(self):
        view_icp_n_matches(nb=self.nb, t=self.tile)

    # icp
    def button_deviations_clicked(self):
        view_icp_deviations(nb=self.nb, t=self.tile)

    #  overlay
    def view_button_clicked(self):
        # This function is called when the view button is clicked
        # Need to get the tile, round, channel and filter from the GUI. Then run view_entire_overlay
        # Get the tile, round, channel and filter from the GUI
        t_view = int(self.overlay_buttons.textbox_tile.text())
        r_view = int(self.overlay_buttons.textbox_round.text())
        c_view = int(self.overlay_buttons.textbox_channel.text())
        filter = self.overlay_buttons.button_filter.isChecked()
        # Run view_entire_overlay.
        try:
            view_entire_overlay(nb=self.nb, t=t_view, r=r_view, c=c_view, filter=filter)
        except:
            print("Error: could not view overlay")
        # Reset view button to unchecked and filter button to unchecked
        self.overlay_buttons.button_overlay.setChecked(False)
        self.overlay_buttons.button_filter.setChecked(False)

    # bg subtraction
    def button_bg_sub_overlay_clicked(self):
        t_view = int(self.bg_sub_buttons.textbox_tile.text())
        r_view = int(self.bg_sub_buttons.textbox_round.text())
        c_view = int(self.bg_sub_buttons.textbox_channel.text())
        # Run view_background_overlay.
        try:
            view_background_overlay(nb=self.nb, t=t_view, r=r_view, c=c_view)
        except:
            print("Error: could not view overlay")
        # Reset view button to unchecked and filter button to unchecked
        self.bg_sub_buttons.button_overlay.setChecked(False)

    # bg subtraction
    def button_brightness_scale_clicked(self):
        t_view = int(self.bg_sub_buttons.textbox_tile.text())
        r_view = int(self.bg_sub_buttons.textbox_round.text())
        c_view = int(self.bg_sub_buttons.textbox_channel.text())
        # Run view_background_overlay.
        try:
            view_background_brightness_correction(nb=self.nb, t=t_view, r=r_view, c=c_view)
        except:
            print("Error: could not view brightness scale")
        # Reset view button to unchecked and filter button to unchecked
        self.bg_sub_buttons.button_brightness_scale.setChecked(False)

    # fluorescent beads
    def button_fluorescent_beads_clicked(self):
        try:
            view_camera_correction(nb=self.nb)
        except:
            print("Error: could not view fluorescent beads")
        # Reset view button to unchecked and filter button to unchecked
        self.bead_buttons.button_fluorescent_beads.setChecked(False)

    # Button functions end here
    def update_plot(self):
        # Updates plot if tile or method has been changed
        # Update the images, we reload the anchor image even when it has not been changed, this should not be too slow
        self.clear_images()
        self.get_images()
        self.plot_images()

    def clear_images(self):
        # Function to clear all images currently in use
        n_images = len(self.viewer.layers)
        for i in range(n_images):
            # when we delete layer 0, layer 1 becomes l0 and so on
            del self.viewer.layers[0]

    def get_images(self):
        # reset initial target image lists to empty lists
        use_rounds = self.nb.basic_info.use_rounds + self.nb.basic_info.use_preseq * [self.nb.basic_info.pre_seq_round]
        use_channels = self.nb.basic_info.use_channels
        self.target_round_image, self.target_channel_image = [], []
        t = self.tile
        # populate target arrays
        for r in use_rounds:
            file = "t" + str(t) + "r" + str(r) + "c" + str(self.round_registration_channel) + ".npy"
            affine = preprocessing.yxz_to_zyx_affine(A=self.transform[t, r, self.c_ref], new_origin=self.new_origin)
            # Reset the spline interpolation order to 1 to speed things up
            self.target_round_image.append(
                affine_transform(np.load(os.path.join(self.output_dir, file)), affine, order=1)
            )

        for c in use_channels:
            file = "t" + str(t) + "r" + str(self.r_mid) + "c" + str(c) + ".npy"
            affine = preprocessing.yxz_to_zyx_affine(A=self.transform[t, self.r_mid, c], new_origin=self.new_origin)
            self.target_channel_image.append(
                affine_transform(np.load(os.path.join(self.output_dir, file)), affine, order=1)
            )
        # populate anchor image
        base_file = "t" + str(t) + "r" + str(self.r_ref) + "c" + str(self.round_registration_channel) + ".npy"
        self.base_image_dapi = np.load(os.path.join(self.output_dir, base_file))
        base_file_anchor = "t" + str(t) + "r" + str(self.r_ref) + "c" + str(self.c_ref) + ".npy"
        self.base_image = np.load(os.path.join(self.output_dir, base_file_anchor))

    def plot_images(self):
        use_rounds = self.nb.basic_info.use_rounds + self.nb.basic_info.use_preseq * [self.nb.basic_info.pre_seq_round]
        use_channels = self.nb.basic_info.use_channels
        n_rounds, n_channels = len(use_rounds), len(use_channels)

        # we have 10 z planes. So we need 10 times as many labels as we have rounds and channels
        features_base_round_reg = {
            "r": np.ones(10 * n_rounds).astype(int) * self.r_ref,
            "c": np.ones(10 * n_rounds).astype(int) * self.round_registration_channel,
        }
        features_target_round_reg = {
            "r": np.repeat(use_rounds, 10).astype(int),
            "c": np.ones(10 * n_rounds).astype(int) * self.round_registration_channel,
        }
        features_base_channel_reg = {
            "r": np.ones(10 * n_channels).astype(int) * self.r_ref,
            "c": np.ones(10 * n_channels).astype(int) * self.c_ref,
        }
        features_target_channel_reg = {
            "r": np.ones(10 * n_channels).astype(int) * self.r_mid,
            "c": np.repeat(use_channels, 10).astype(int),
        }

        # Define text
        text_base = {"string": "R: {r} C: {c}", "size": 8, "color": "red"}
        text_target = {"string": "R: {r} C: {c}", "size": 8, "color": "green"}

        # Now go on to define point coords. Napari only allows us to plot text with points, so will plot points that
        # are not visible and attach text to them
        round_reg_points = []
        channel_reg_points = []

        for r in range(n_rounds):
            self.viewer.add_image(
                self.base_image_dapi, blending="additive", colormap="red", translate=[0, 0, 1_000 * r]
            )
            self.viewer.add_image(
                self.target_round_image[r], blending="additive", colormap="green", translate=[0, 0, 1_000 * r]
            )
            # Add this to all z planes so still shows up when scrolling
            for z in range(10):
                round_reg_points.append([z, -50, 250 + 1_000 * r])

        for c in range(n_channels):
            self.viewer.add_image(self.base_image, blending="additive", colormap="red", translate=[0, 1_000, 1_000 * c])
            self.viewer.add_image(
                self.target_channel_image[c], blending="additive", colormap="green", translate=[0, 1_000, 1_000 * c]
            )
            for z in range(10):
                channel_reg_points.append([z, 950, 250 + 1_000 * c])

        round_reg_points, channel_reg_points = np.array(round_reg_points), np.array(channel_reg_points)

        # Add text to image
        anchor_offset = np.array([0, 100, 0])
        self.viewer.add_points(
            round_reg_points - anchor_offset, features=features_base_round_reg, text=text_base, size=0
        )
        self.viewer.add_points(round_reg_points, features=features_target_round_reg, text=text_target, size=0)
        self.viewer.add_points(
            channel_reg_points - anchor_offset, features=features_base_channel_reg, text=text_base, size=0
        )
        self.viewer.add_points(channel_reg_points, features=features_target_channel_reg, text=text_target, size=0)


class ButtonMethodWindow(QMainWindow):
    def __init__(self, active_button: str = "SVR"):
        super().__init__()
        self.button_svr = QPushButton("SVR", self)
        self.button_svr.setCheckable(True)
        self.button_svr.setGeometry(75, 2, 50, 28)  # left, top, width, height

        self.button_icp = QPushButton("ICP", self)
        self.button_icp.setCheckable(True)
        self.button_icp.setGeometry(140, 2, 50, 28)  # left, top, width, height
        if active_button.lower() == "icp":
            # Initially, show sub vol regression registration
            self.button_icp.setChecked(True)
            self.method = "ICP"
        elif active_button.lower() == "svr":
            self.button_svr.setChecked(True)
            self.method = "SVR"
        else:
            raise ValueError(f"active_button should be 'SVR' or 'ICP' but {active_button} was given.")


class ButtonOnOffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button_base = QPushButton("Base", self)
        self.button_base.setCheckable(True)
        self.button_base.setGeometry(75, 2, 50, 28)

        self.button_target = QPushButton("Target", self)
        self.button_target.setCheckable(True)
        self.button_target.setGeometry(140, 2, 50, 28)


class ButtonTileWindow(QMainWindow):
    def __init__(self, tile_pos_xy: np.ndarray, use_tiles: list, active_button: 0):
        super().__init__()
        # Loop through tiles, putting them in location as specified by tile pos xy
        for t in range(len(tile_pos_xy)):
            # Create a button for each tile
            button = QPushButton(str(t), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(t in use_tiles)
            button.setGeometry(tile_pos_xy[t, 0] * 70, tile_pos_xy[t, 1] * 40, 50, 28)
            # set active button as checked
            if active_button == t:
                button.setChecked(True)
                self.tile = t
            # Set button color = grey when hovering over
            # set colour of tiles in use to blue amd not in use to red
            if t in use_tiles:
                button.setStyleSheet(
                    "QPushButton"
                    "{"
                    "background-color : rgb(135, 206, 250);"
                    "}"
                    "QPushButton::hover"
                    "{"
                    "background-color : lightgrey;"
                    "}"
                    "QPushButton::pressed"
                    "{"
                    "background-color : white;"
                    "}"
                )
            else:
                button.setStyleSheet(
                    "QPushButton"
                    "{"
                    "background-color : rgb(240, 128, 128);"
                    "}"
                    "QPushButton::hover"
                    "{"
                    "background-color : lightgrey;"
                    "}"
                    "QPushButton::pressed"
                    "{"
                    "background-color : white;"
                    "}"
                )
            # Finally add this button as an attribute to self
            self.__setattr__(str(t), button)


class ButtonSVRWindow(QMainWindow):
    # This class creates a window with buttons for all SVR diagnostics
    # This includes buttons for each round and channel regression
    # Also includes a button to view pearson correlation coefficient in either a histogram or colormap
    # Also includes a button to view pearson correlation coefficient spatially for either rounds or channels
    def __init__(self, use_rounds: list, use_channels: list):
        super().__init__()
        # Create round regression buttons
        for r in use_rounds:
            # Create a button for each tile
            button = QPushButton("R" + str(r), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            x, y_r = r % 4, r // 4
            button.setGeometry(x * 70, 40 + 60 * y_r, 50, 28)
            # Finally add this button as an attribute to self
            self.__setattr__("R" + str(r), button)

        # Create 2 correlation buttons:
        # 1 to view pearson correlation coefficient as histogram
        # 2 to view pearson correlation coefficient as colormap
        y = y_r + 1
        button = QPushButton("Shift Score \n Hist", self)
        button.setCheckable(True)
        button.setGeometry(0, 40 + 60 * y, 120, 56)
        self.pearson_hist = button
        button = QPushButton("Shift Score \n c-map", self)
        button.setCheckable(True)
        button.setGeometry(140, 40 + 60 * y, 120, 56)
        self.pearson_cmap = button

        # Create 2 spatial correlation buttons:
        # 1 to view pearson correlation coefficient spatially for rounds
        # 2 to view pearson correlation coefficient spatially for channels
        y += 1
        button = QPushButton("Round Score \n Spatial c-map", self)
        button.setCheckable(True)
        button.setGeometry(0, 68 + 60 * y, 120, 56)
        self.pearson_spatial_round = button


class ButtonFlowWindow(QMainWindow):
    # This class creates a window with buttons for all Flow diagnostics
    # This includes a button for each round optical flow (including preseq if it exists)
    def __init__(self, use_rounds: list):
        super().__init__()
        # Create round regression buttons
        for r in use_rounds:
            # Create a button for each tile
            button = QPushButton('R' + str(r), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            x, y_r = r % 4, r // 4
            button.setGeometry(x * 70, 40 + 60 * y_r, 50, 28)
            # Finally add this button as an attribute to self
            self.__setattr__('R' + str(r), button)


class ButtonOutlierWindow(QMainWindow):
    # This class creates a window with buttons for all outlier removal diagnostics
    # This includes round and channel button to view shifts for each tile as a vector field
    # Also includes round and channel button to view shifts for each tile as a heatmap
    # Also includes round and channel button to view boxplots of scales for each tile
    def __init__(self):
        super().__init__()
        self.button_vec_field_r = QPushButton("Round Shift Vector Field", self)
        self.button_vec_field_r.setCheckable(True)
        self.button_vec_field_r.setGeometry(20, 40, 220, 28)

        self.button_vec_field_c = QPushButton("Channel Shift Vector Field", self)
        self.button_vec_field_c.setCheckable(True)
        self.button_vec_field_c.setGeometry(20, 100, 220, 28)  # left, top, width, height

        self.button_shift_cmap_r = QPushButton("Round Shift Colour Map", self)
        self.button_shift_cmap_r.setCheckable(True)
        self.button_shift_cmap_r.setGeometry(20, 160, 220, 28)

        self.button_shift_cmap_c = QPushButton("Channel Shift Colour Map", self)
        self.button_shift_cmap_c.setCheckable(True)
        self.button_shift_cmap_c.setGeometry(20, 220, 220, 28)  # left, top, width, height

        self.button_scale_r = QPushButton("Round Scales", self)
        self.button_scale_r.setCheckable(True)
        self.button_scale_r.setGeometry(20, 280, 100, 28)

        self.button_scale_c = QPushButton("Channel Scales", self)
        self.button_scale_c.setCheckable(True)
        self.button_scale_c.setGeometry(140, 280, 100, 28)  # left, top, width, height


class ButtonICPWindow(QMainWindow):
    # This class creates a window with buttons for all ICP diagnostics
    # One diagnostic for MSE, one for n_matches, one for icp_deciations
    def __init__(self):
        super().__init__()
        self.button_mse = QPushButton("MSE", self)
        self.button_mse.setCheckable(True)
        self.button_mse.setGeometry(20, 40, 100, 28)

        self.button_matches = QPushButton("Matches", self)
        self.button_matches.setCheckable(True)
        self.button_matches.setGeometry(140, 40, 100, 28)  # left, top, width, height

        self.button_deviations = QPushButton("Large ICP Deviations", self)
        self.button_deviations.setCheckable(True)
        self.button_deviations.setGeometry(20, 100, 220, 28)


class ButtonOverlayWindow(QMainWindow):
    # This class creates a window with buttons for viewing overlays
    # We want text boxes for entering the tile, round and channel. We then want a simple button for filtering and
    # for viewing the overlay
    def __init__(self):
        super().__init__()

        self.button_overlay = QPushButton("View", self)
        self.button_overlay.setCheckable(True)
        self.button_overlay.setGeometry(20, 20, 220, 28)

        # Add title to each textbox
        label_tile = QLabel(self)
        label_tile.setText("Tile")
        label_tile.setGeometry(20, 70, 100, 28)
        self.textbox_tile = QLineEdit(self)
        self.textbox_tile.setFont(QFont("Arial", 8))
        self.textbox_tile.setText("0")
        self.textbox_tile.setGeometry(20, 100, 100, 28)

        label_round = QLabel(self)
        label_round.setText("Round")
        label_round.setGeometry(140, 70, 100, 28)
        self.textbox_round = QLineEdit(self)
        self.textbox_round.setFont(QFont("Arial", 8))
        self.textbox_round.setText("0")
        self.textbox_round.setGeometry(140, 100, 100, 28)

        label_channel = QLabel(self)
        label_channel.setText("Channel")
        label_channel.setGeometry(20, 130, 100, 28)
        self.textbox_channel = QLineEdit(self)
        self.textbox_channel.setFont(QFont("Arial", 8))
        self.textbox_channel.setText("18")
        self.textbox_channel.setGeometry(20, 160, 100, 28)

        self.button_filter = QPushButton("Filter", self)
        self.button_filter.setCheckable(True)
        self.button_filter.setGeometry(140, 160, 100, 28)


class ButtonBGWindow(QMainWindow):
    """
    This class creates a window with buttons for viewing background images overlayed with foreground images
    """

    def __init__(self):
        super().__init__()
        self.button_overlay = QPushButton("View Overlay", self)
        self.button_overlay.setCheckable(True)
        self.button_overlay.setGeometry(20, 20, 220, 28)

        self.button_brightness_scale = QPushButton("View BG Scale", self)
        self.button_brightness_scale.setCheckable(True)
        self.button_brightness_scale.setGeometry(20, 70, 220, 28)

        # Add title to each textbox
        label_tile = QLabel(self)
        label_tile.setText("Tile")
        label_tile.setGeometry(20, 130, 100, 28)
        self.textbox_tile = QLineEdit(self)
        self.textbox_tile.setFont(QFont("Arial", 8))
        self.textbox_tile.setText("0")
        self.textbox_tile.setGeometry(20, 160, 100, 28)

        label_round = QLabel(self)
        label_round.setText("Round")
        label_round.setGeometry(140, 130, 100, 28)
        self.textbox_round = QLineEdit(self)
        self.textbox_round.setFont(QFont("Arial", 8))
        self.textbox_round.setText("0")
        self.textbox_round.setGeometry(140, 160, 100, 28)

        label_channel = QLabel(self)
        label_channel.setText("Channel")
        label_channel.setGeometry(20, 190, 100, 28)
        self.textbox_channel = QLineEdit(self)
        self.textbox_channel.setFont(QFont("Arial", 8))
        self.textbox_channel.setText("18")
        self.textbox_channel.setGeometry(20, 220, 100, 28)


class ButtonBeadWindow(QMainWindow):
    """
    This class creates a window with buttons for viewing fluorescent bead images
    """

    def __init__(self):
        super().__init__()
        self.button_fluorescent_beads = QPushButton("View Fluorescent Beads", self)
        self.button_fluorescent_beads.setCheckable(True)
        self.button_fluorescent_beads.setGeometry(20, 20, 220, 28)


def set_style(button):
    # Set button color = grey when hovering over, blue when pressed, white when not
    button.setStyleSheet(
        "QPushButton"
        "{"
        "background-color : rgb(135, 206, 250);"
        "}"
        "QPushButton::hover"
        "{"
        "background-color : lightgrey;"
        "}"
        "QPushButton::pressed"
        "{"
        "background-color : white;"
        "}"
    )
    return button


# 1
def view_round_regression_scatter(nb: Notebook, t: int, r: int):
    """
    view 9 scatter plots for each data set shift vs positions
    Args:
        nb: Notebook
        t: tile
        r: round
    """
    # Transpose shift and position variables so coord is dimension 0, makes plotting easier
    shift = nb.register_debug.round_shift[t, r]
    corr = nb.register_debug.round_shift_corr[t, r]
    position = nb.register_debug.position
    initial_transform = nb.register_debug.round_transform_raw[t, r]
    icp_transform = preprocessing.yxz_to_zyx_affine(A=nb.register.transform[t, r, nb.basic_info.anchor_channel])

    r_thresh = nb.get_config()["register"]["pearson_r_thresh"]
    shift = shift[corr > r_thresh].T
    position = position[corr > r_thresh].T

    # Make ranges, wil be useful for plotting lines
    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))
    coord_range = [z_range, yx_range, yx_range]
    # Need to add a central offset to all lines plotted
    tile_centre_zyx = np.roll(nb.basic_info.tile_centre, 1)

    # We want to plot the shift of each coord against the position of each coord. The gradient when the dependent var
    # is coord i and the independent var is coord j should be the transform[i,j] - int(i==j)
    gradient_svr = initial_transform[:3, :3] - np.eye(3)
    gradient_icp = icp_transform[:3, :3] - np.eye(3)
    # Now we need to compute what the intercept should be for each coord. Usually this would just be given by the final
    # column of the transform, but we need to add a central offset to this. If the dependent var is coord i, and the
    # independent var is coord j, then the intercept should be the transform[i,3] + central_offset[i,j]. This central
    # offset is given by the formula: central_offset[i, j] = gradient[i, k1] * tile_centre[k1] + gradient[i, k2] *
    # tile_centre[k2], where k1 and k2 are the coords that are not j.
    central_offset_svr = np.zeros((3, 3))
    central_offset_icp = np.zeros((3, 3))
    intercpet_svr = np.zeros((3, 3))
    intercpet_icp = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # k1 and k2 are the coords that are not j
            k1 = (j + 1) % 3
            k2 = (j + 2) % 3
            central_offset_svr[i, j] = (
                gradient_svr[i, k1] * tile_centre_zyx[k1] + gradient_svr[i, k2] * tile_centre_zyx[k2]
            )
            central_offset_icp[i, j] = (
                gradient_icp[i, k1] * tile_centre_zyx[k1] + gradient_icp[i, k2] * tile_centre_zyx[k2]
            )
            # Now compute the intercepts
            intercpet_svr[i, j] = initial_transform[i, 3] + central_offset_svr[i, j]
            intercpet_icp[i, j] = icp_transform[i, 3] + central_offset_icp[i, j]

    # Define the axes
    fig, axes = plt.subplots(3, 3)
    coord = ["Z", "Y", "X"]
    # Now plot n_matches
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(x=position[j], y=shift[i], alpha=0.3)
            ax.plot(coord_range[j], gradient_svr[i, j] * coord_range[j] + intercpet_svr[i, j], label="SVR")
            ax.plot(coord_range[j], gradient_icp[i, j] * coord_range[j] + intercpet_icp[i, j], label="ICP")
            ax.legend()
    # Label subplot rows and columns with coord names
    for ax, col in zip(axes[0], coord):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], coord):
        ax.set_ylabel(row, rotation=90, size="large")
    # common axis labels
    fig.supxlabel("Position")
    fig.supylabel("Shift")
    # Add title
    round_registration_channel = nb.get_config()["register"]["round_registration_channel"]
    if round_registration_channel is None:
        round_registration_channel = nb.basic_info.anchor_channel
    plt.suptitle(
        "Round regression for Tile " + str(t) + ", Round " + str(r) + "Channel " + str(round_registration_channel)
    )
    plt.show()


# 1
def view_pearson_hists(nb, t, num_bins=30):
    """
    function to view histogram of correlation coefficients for all subvol shifts of all round/channels.
    Args:
        nb: Notebook
        t: int tile under consideration
        num_bins: int number of bins in the histogram
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    thresh = nb.get_config()["register"]["pearson_r_thresh"]
    round_corr = nbp_register_debug.round_shift_corr[t]
    n_rounds = nbp_basic.n_rounds
    cols = n_rounds

    for r in range(n_rounds):
        plt.subplot(1, cols, r + 1)
        counts, _ = np.histogram(round_corr[r], np.linspace(0, 1, num_bins))
        plt.hist(round_corr[r], bins=np.linspace(0, 1, num_bins))
        plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors="r")
        # change fontsize from default 10 to 7
        plt.title(
            "r = "
            + str(r)
            + "\n Pass = "
            + str(round(100 * sum(round_corr[r] > thresh) / round_corr.shape[1], 2))
            + "%",
            fontsize=7,
        )
        # remove x ticks and y ticks
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Similarity Score Distributions for all Sub-Volume Shifts")
    plt.show()


# 1
def view_pearson_colourmap(nb, t):
    """
    function to view colourmap of correlation coefficients for all subvol shifts for all channels and rounds.

    Args:
        nb: Notebook
        t: int tile under consideration
    """
    # initialise frequently used variables
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    round_corr = nbp_register_debug.round_shift_corr[t]

    # Replace 0 with nans so they get plotted as black
    round_corr[round_corr == 0] = np.nan
    # plot round correlation
    fig, ax = plt.subplots(1, 1)
    # ax1 refers to round shifts
    im = ax.imshow(round_corr, vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax.set_xlabel("Sub-volume index")
    ax.set_ylabel("Round")
    ax.set_title("Round sub-volume shift scores")

    # Add common colour bar. Also give it the label 'Pearson correlation coefficient'
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Correlation coefficient")

    plt.suptitle("Similarity score distributions for all sub-volume shifts")


# 1
def view_pearson_colourmap_spatial(nb: Notebook, t: int):
    """
    function to view colourmap of correlation coefficients along with spatial info for either all round shifts of a tile
    or all channel shifts of a tile.

    Args:
        nb: Notebook
        round: True if round, false if channel
        t: tile under consideration
    """

    # initialise frequently used variables
    config = nb.get_config()["register"]
    use = nb.basic_info.use_rounds
    corr = nb.register_debug.round_shift_corr[t, use]
    mode = "Round"

    # Set 0 correlations to nan, so they are plotted as black
    corr[corr == 0] = np.nan
    z_subvols, y_subvols, x_subvols = config["subvols"]
    n_rc = corr.shape[0]

    fig, axes = plt.subplots(nrows=z_subvols, ncols=n_rc)
    if axes.ndim == 1:
        axes = axes[None]
    # Now plot each image
    for elem in range(n_rc):
        for z in range(z_subvols):
            ax = axes[z, elem]
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(
                np.reshape(
                    corr[elem, z * y_subvols * x_subvols : (z + 1) * x_subvols * y_subvols], (y_subvols, x_subvols)
                ),
                vmin=0,
                vmax=1,
            )
    # common axis labels
    fig.supxlabel(mode)
    fig.supylabel("Z-Subvolume")
    # Set row and column labels
    for ax, col in zip(axes[0], use):
        ax.set_title(col, size="large")
    for ax, row in zip(axes[:, 0], np.arange(z_subvols)):
        ax.set_ylabel(row, rotation=0, size="large", x=-0.1)
    # add colour bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Correlation coefficient")

    plt.suptitle(mode + " shift similarity scores for tile " + str(t) + " plotted spatially")


# 2
def shift_vector_field(nb: Notebook):
    """
    Function to plot vector fields of predicted shifts vs shifts to see if we classify a shift as an outlier.
    Args:
        nb: Notebook
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    residual_thresh = nb.get_config()["register"]["residual_thresh"]
    use_tiles = nbp_basic.use_tiles
    shift = nbp_register_debug.round_transform_raw[use_tiles, :, :, 3]
    # record number of rounds, tiles and initialise predicted shift
    n_tiles, n_rounds = shift.shape[0], nbp_basic.n_rounds
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]
    tilepos_yx_padded = np.vstack((tilepos_yx.T, np.ones(n_tiles))).T
    predicted_shift = np.zeros_like(shift)
    # When we are scaling the vector field, it will be useful to store the following
    n_vectors_x = tilepos_yx[:, 1].max() - tilepos_yx[:, 1].min() + 1
    shift_norm = np.linalg.norm(shift, axis=2)

    fig, axes = plt.subplots(nrows=3, ncols=n_rounds)
    for r in range(n_rounds):
        # generate predicted shift for this round
        lb, ub = np.percentile(shift_norm[:, r], [10, 90])
        valid = (shift_norm[:, r] > lb) * (shift_norm[:, r] < ub)
        # Carry out regression, first predicitng yx shift, then z shift
        transform_yx = np.linalg.lstsq(tilepos_yx_padded[valid], shift[valid, r, 1:], rcond=None)[0]
        predicted_shift[:, r, 1:] = tilepos_yx_padded @ transform_yx
        transform_z = np.linalg.lstsq(tilepos_yx_padded[valid], shift[valid, r, 0][:, None], rcond=None)[0]
        predicted_shift[:, r, 0] = (tilepos_yx_padded @ transform_z)[:, 0]
        # Defining this scale will mean that the length of the largest vector will be equal to 1/n_vectors_x of the
        # width of the plot
        scale = n_vectors_x * np.sqrt(np.sum(predicted_shift[:, r, 1:] ** 2, axis=1))

        # plot the predicted yx shift vs actual yx shift in row 0
        ax = axes[0, r]
        # Make sure the vector field is properly scaled
        ax.quiver(
            tilepos_yx[:, 1],
            tilepos_yx[:, 0],
            predicted_shift[:, r, 2],
            predicted_shift[:, r, 1],
            color="b",
            scale=scale,
            scale_units="width",
            width=0.05,
            alpha=0.5,
        )
        ax.quiver(
            tilepos_yx[:, 1],
            tilepos_yx[:, 0],
            shift[:, r, 2],
            shift[:, r, 1],
            color="r",
            scale=scale,
            scale_units="width",
            width=0.05,
            alpha=0.5,
        )
        # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
        ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
        ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # plot the predicted z shift vs actual z shift in row 1
        ax = axes[1, r]
        # we only want 1 label so make this for r = 0
        if r == 0:
            ax.quiver(
                tilepos_yx[:, 1],
                tilepos_yx[:, 0],
                0,
                predicted_shift[:, r, 0],
                color="b",
                scale=scale,
                scale_units="width",
                width=0.05,
                alpha=0.5,
                label="Predicted",
            )
            ax.quiver(
                tilepos_yx[:, 1],
                tilepos_yx[:, 0],
                0,
                shift[:, r, 2],
                color="r",
                scale=scale,
                scale_units="width",
                width=0.05,
                alpha=0.5,
                label="Actual",
            )
        else:
            ax.quiver(
                tilepos_yx[:, 1],
                tilepos_yx[:, 0],
                0,
                predicted_shift[:, r, 0],
                color="b",
                scale=scale,
                scale_units="width",
                width=0.05,
                alpha=0.5,
            )
            ax.quiver(
                tilepos_yx[:, 1],
                tilepos_yx[:, 0],
                0,
                shift[:, r, 2],
                color="r",
                scale=scale,
                scale_units="width",
                width=0.05,
                alpha=0.5,
            )
        # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
        ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
        ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot image of norms of residuals at each tile in row 3
        ax = axes[2, r]
        diff = create_tiled_image(data=np.linalg.norm(predicted_shift[:, r] - shift[:, r], axis=1), nbp_basic=nbp_basic)
        outlier = np.argwhere(diff > residual_thresh)
        n_outliers = outlier.shape[0]
        im = ax.imshow(diff, vmin=0, vmax=10)
        # Now we want to outline the outlier pixels with a dotted red rectangle
        for i in range(n_outliers):
            rect = patches.Rectangle(
                (outlier[i, 1] - 0.5, outlier[i, 0] - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])

    # Set row and column labels
    for ax, col in zip(axes[0], nbp_basic.use_rounds):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], ["XY-shifts", "Z-shifts", "Residuals"]):
        ax.set_ylabel(row, rotation=90, size="large")
    # Add row and column labels
    fig.supxlabel("Round")
    fig.supylabel("Diagnostic")
    # Add global colour bar and legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Residual Norm")
    # Add title
    fig.suptitle("Diagnostic plots for round shift outlier removal", size="x-large")


# 2
def zyx_shift_image(nb: Notebook, round: bool = True):
    """
    Function to plot overlaid images of predicted shifts vs shifts to see if we classify a shift as an outlier.
    Args:
        nb: Notebook
        round: Boolean indicating whether we are looking at round outlier removal, True if r, False if c
    """
    nbp_basic, nbp_register, nbp_register_debug = nb.basic_info, nb.register, nb.register_debug
    use_tiles = nbp_basic.use_tiles

    # Load in shift
    if round:
        mode = "Round"
        use = nbp_basic.use_rounds
        shift_raw = nbp_register_debug.round_transform_raw[use_tiles, :, :, 3]
        shift = nbp_register.round_transform[use_tiles, :, :, 3]
    else:
        mode = "Channel"
        use = nbp_basic.use_channels
        shift_raw = nbp_register_debug.channel_transform_raw[use_tiles, :, :, 3]
        shift = nbp_register.channel_transform[use_tiles, :, :, 3]

    coord_label = ["Z", "Y", "X"]
    n_t, n_rc = shift.shape[0], shift.shape[1]
    fig, axes = plt.subplots(nrows=3, ncols=n_rc)
    # common axis labels
    fig.supxlabel(mode)
    fig.supylabel("Coordinate (Z, Y, X)")

    # Set row and column labels
    for ax, col in zip(axes[0], use):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], coord_label):
        ax.set_ylabel(row, rotation=0, size="large")

    # Now we will plot 3 rows of subplots and n_rc columns of subplots. Each subplot will be made up of 2 further subplots
    # The Left subplot will be the raw shift and the right will be the regularised shift
    # We will also outline pixels in these images that are different between raw and regularised with a dotted red rectangle
    for elem in range(n_rc):
        for coord in range(3):
            ax = axes[coord, elem]
            # remove the ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Create 2 subplots within each subplot
            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax, wspace=0.1)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            # Plot the raw shift in the left subplot
            im = ax1.imshow(create_tiled_image(shift_raw[:, elem, coord], nbp_basic))
            # Plot the regularised shift in the right subplot
            im = ax2.imshow(create_tiled_image(shift[:, elem, coord], nbp_basic))
            # Now we want to outline the pixels that are different between raw and regularised with a dotted red rectangle
            diff = np.abs(shift_raw[:, elem, coord] - shift[:, elem, coord])
            outlier = np.argwhere(diff > 0.1)
            n_outliers = outlier.shape[0]
            for i in range(n_outliers):
                rect = patches.Rectangle(
                    (outlier[i, 1] - 0.5, outlier[i, 0] - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    linestyle="--",
                )
                # Add the rectangle to both subplots
                ax1.add_patch(rect)
                ax2.add_patch(rect)
                # Remove ticks and labels from the left subplot
                ax1.set_xticks([])
                ax1.set_yticks([])
                # Remove ticks and labels from the right subplot
                ax2.set_xticks([])
                ax2.set_yticks([])

    fig.canvas.draw()
    plt.show()
    # Add a title
    fig.suptitle("Diagnostic plots for {} shift outlier removal".format(mode), size="x-large")
    # add a global colour bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


# 2
def create_tiled_image(data, nbp_basic):
    """
    generate image of 1d tile data along with tile positions
    Args:
        data: n_tiles_use x 1 list of residuals
        nbp_basic: basic info notebook page
    """
    # Initialise frequently used variables
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx[nbp_basic.use_tiles]
    n_rows = np.max(tilepos_yx[:, 0]) - np.min(tilepos_yx[:, 0]) + 1
    n_cols = np.max(tilepos_yx[:, 1]) - np.min(tilepos_yx[:, 1]) + 1
    tilepos_yx = tilepos_yx - np.min(tilepos_yx, axis=0)
    diff = np.zeros((n_rows, n_cols))

    for t in range(len(use_tiles)):
        diff[tilepos_yx[t, 0], tilepos_yx[t, 1]] = data[t]

    diff[diff == 0] = np.nan

    return diff

# 3
# TODO: Update this so it uses icp transforms...
# def view_channel_scales(nb: Notebook):
#     """
#     view scale parameters for the round outlier removals
#     Args:
#         nb: Notebook
#     """
#     nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
#     mid_round, anchor_channel = nbp_basic.n_rounds // 2, nbp_basic.anchor_channel
#     use_tiles = nbp_basic.use_tiles
#     use_channels = nbp_basic.use_channels
#     # Extract raw scales
#     z_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 0, 0]
#     y_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 1, 1]
#     x_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 2, 2]
#     n_tiles_use, n_channels_use = z_scale.shape[0], z_scale.shape[1]
#
#     # Plot box plots
#     plt.subplot(3, 1, 1)
#     plt.scatter(np.tile(np.arange(n_channels_use), n_tiles_use), np.reshape(z_scale, (n_tiles_use * n_channels_use)),
#                 c='w', marker='x')
#     plt.plot(np.arange(n_channels_use), 0.99 * np.ones(n_channels_use), 'c:', label='0.99 - 1.01')
#     plt.plot(np.arange(n_channels_use), np.ones(n_channels_use), 'r:', label='1')
#     plt.plot(np.arange(n_channels_use), 1.01 * np.ones(n_channels_use), 'c:')
#     plt.xticks(np.arange(n_channels_use), use_channels)
#     plt.xlabel('Channel')
#     plt.ylabel('Z-scale')
#     plt.legend()
#
#     plt.subplot(3, 1, 2)
#     plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
#                 y=np.reshape(y_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
#     plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 25, axis=0), 'c:', label='Inter Quartile Range')
#     plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 50, axis=0), 'r:', label='Median')
#     plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 75, axis=0), 'c:')
#     plt.xticks(np.arange(n_channels_use), use_channels)
#     plt.xlabel('Channel')
#     plt.ylabel('Y-scale')
#     plt.legend()
#
#     plt.subplot(3, 1, 3)
#     plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
#                 y=np.reshape(x_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
#     plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 25, axis=0), 'c:', label='Inter Quartile Range')
#     plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 50, axis=0), 'r:', label='Median')
#     plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 75, axis=0), 'c:')
#     plt.xticks(np.arange(n_channels_use), use_channels)
#     plt.xlabel('Channel')
#     plt.ylabel('X-scale')
#     plt.legend()
#
#     plt.suptitle('Distribution of scales across tiles for channel registration.')
#     plt.show()
# 3
def view_icp_n_matches(nb: Notebook, t: int):
    """
    Plots simple proportion matches against iterations.
    Args:
        nb: Notebook
        t: tile
    """
    nbp_basic, nbp_register_debug, nbp_find_spots = nb.basic_info, nb.register_debug, nb.find_spots
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_matches = nbp_register_debug.n_matches[t, use_rounds][:, use_channels]
    # delete column 1 from n_matches as it is incorrect
    n_matches = np.delete(n_matches, 1, axis=2)
    frac_matches = preprocessing.n_matches_to_frac_matches(
        n_matches=n_matches, spot_no=nbp_find_spots.spot_no[t, use_rounds][:, use_channels]
    )
    n_iters = n_matches.shape[2]

    # Define the axes
    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    # common axis labels
    fig.supxlabel("Channels")
    fig.supylabel("Rounds")
    # Set row and column labels
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], use_rounds):
        ax.set_ylabel(row, rotation=0, size="large")

    # Now plot n_matches
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            ax.plot(np.arange(n_iters), frac_matches[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim([0, 1])
            ax.set_xlim([0, n_iters // 2])

    plt.suptitle("Fraction of matches against iterations for tile " + str(t) + ". \n " "Note that y-axis is [0,1]")
    plt.show()


# 3
def view_icp_mse(nb: Notebook, t: int):
    """
    Plots simple MSE grid against iterations
    Args:
        nb: Notebook
        t: tile
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    mse = nbp_register_debug.mse[t, use_rounds][:, use_channels]
    # delete column 1 from mse as it is incorrect
    mse = np.delete(mse, 1, axis=2)
    n_iters = mse.shape[2]

    # Define the axes
    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    # common axis labels
    fig.supxlabel("Channels")
    fig.supylabel("Rounds")
    # Set row and column labels
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], use_rounds):
        ax.set_ylabel(row, rotation=0, size="large")

    # Now plot mse
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            ax.plot(np.arange(n_iters), mse[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, n_iters // 2])
            ax.set_ylim([0, np.max(mse)])

    plt.suptitle(
        "MSE against iteration for tile " + str(t) + " for all rounds and channels. \n"
        "Note that the y-axis is the same for all plots."
    )
    plt.show()


# TODO: This will need to be refactored. We will only need to compare ICP correction to identity transform
# 3
# def view_icp_deviations(nb: Notebook, t: int):
#     """
#     Plots deviations of ICP transform for a given tile t (n_rounds x n_channel x 3 x 4) affine transform against initial
#     guess (initial_transform) which has the same shape. These trasnforms are in zyx x zyx format, with the final col
#     referring to the shift. Our plot has rows as rounds and columns as channels, giving us len(use_rounds) rows, and
#     len(use_channels) columns of subplots.
#
#     Each subplot will be a 2 3x1 images where the first im is [z_scale_icp - z_scale_svr, y_scale_icp - y_scale_svr,
#     x_scale_icp - x_scale_svr], and second im is [z_shift_icp - z_shift_svr, y_shift_icp - y_shift_svr,
#     s_shift_icp - x_shift_svr]. There should be a common colour bar on the right for all scale difference images and
#     another on the right for all shift difference images.
#
#     Args:
#         nb: Notebook
#         t: tile
#     """
#     # Initialise frequent variables
#     nbp_basic, nbp_register = nb.basic_info, nb.register
#     use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
#     initial_transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
#     transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
#     for r in range(len(use_rounds)):
#         for c in range(len(use_channels)):
#             initial_transform[r, c] = preprocessing.yxz_to_zyx_affine(
#                 A=nbp_register.initial_transform[t, use_rounds[r], use_channels[c]])
#             transform[r, c] = preprocessing.yxz_to_zyx_affine(A=nbp_register.transform[t, use_rounds[r], use_channels[c]])
#
#     # Define the axes
#     fig, axes = plt.subplots(len(use_rounds), len(use_channels))
#     # common axis labels
#     fig.supxlabel('Channels')
#     fig.supylabel('Rounds')
#     # Set row and column labels
#     for ax, col in zip(axes[0], use_channels):
#         ax.set_title(col)
#     for ax, row in zip(axes[:, 0], use_rounds):
#         ax.set_ylabel(row, rotation=0, size='large')
#
#     # Define difference images
#     scale_diff = np.zeros((len(use_rounds), len(use_channels), 3))
#     shift_diff = np.zeros((len(use_rounds), len(use_channels), 3))
#     for r in range(len(use_rounds)):
#         for c in range(len(use_channels)):
#             scale_diff[r, c] = np.diag(transform[r, c, :3, :3]) - np.diag(initial_transform[r, c, :3, :3])
#             shift_diff[r, c] = transform[r, c, :3, 3] - initial_transform[r, c, :3, 3]
#
#     # Now plot scale_diff
#     for r in range(len(use_rounds)):
#         for c in range(len(use_channels)):
#             ax = axes[r, c]
#             # create 2 subplots within this subplot
#             ax1 = ax.inset_axes([0, 0, 0.5, 1])
#             ax2 = ax.inset_axes([0.5, 0, 0.5, 1])
#             # plot scale_diff
#             im1 = ax1.imshow(scale_diff[r, c].reshape(3, 1), cmap='bwr', vmin=-.1, vmax=.1)
#             # plot shift_diff
#             im2 = ax2.imshow(shift_diff[r, c].reshape(3, 1), cmap='bwr', vmin=-5, vmax=5)
#             # remove ticks
#             ax1.set_xticks([])
#             ax1.set_yticks([])
#             ax2.set_xticks([])
#             ax2.set_yticks([])
#             ax.set_xticks([])
#             ax.set_yticks([])
#
#     # plot 2 colour bars, one for the shift_diff and one for the scale_diff. Both colour bars should be the same size,
#     # and the scale_diff colour bar should be on the left of the subplots, and the shift_diff colour bar should be on
#     # the right of the subplots.
#     fig.subplots_adjust(right=0.7)
#     cbar_scale_ax = fig.add_axes([0.75, 0.15, 0.025, 0.7])
#     # Next we want to make sure the scale cbar has ticks on the left.
#     cbar_scale_ax.yaxis.tick_left()
#     fig.colorbar(im1, cax=cbar_scale_ax, ticks=[-.1, 0, .1], label='Scale difference')
#     cbar_shift_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
#     fig.colorbar(im2, cax=cbar_shift_ax, ticks=[-5, 0, 5], label='Shift difference')
#
#     plt.suptitle('Deviations of ICP from SVR for tile ' + str(t) + '. \n'
#                                                                    'Left column is zyx scale difference, '
#                                                                    'right column is zyx shift difference.')


# TODO: This will need to be refactored. We will need to use flow instead of affine transforms
# def view_entire_overlay(nb: Notebook, t: int, r: int, c: int, filter=False, initial=False):
#     """
#     Plots the entire image for a given tile t, round r, channel c, and overlays the SVR transformed image on top of the
#     ICP transformed image. The SVR transformed image is in red, and the ICP transformed image is in green.
#
#     Args:
#         nb: Notebook
#         t: tile
#         r: round
#         c: channel
#         filter: whether to apply sobel filter to images
#         initial: whether to use initial transform or final transform
#     """
#     # Initialise frequent variables
#     anchor = preprocessing.yxz_to_zyx(tiles_io.load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t,
#                                                           nb.basic_info.anchor_round, nb.basic_info.anchor_channel,
#                                                           apply_shift=False))
#     anchor = anchor.astype(np.int32)
#     anchor -= nb.basic_info.tile_pixel_value_shift
#     anchor = anchor.astype(np.float32)
#     target = preprocessing.yxz_to_zyx(tiles_io.load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t, r, c,
#                                                           apply_shift=False))
#     target = target.astype(np.int32)
#     target -= nb.basic_info.tile_pixel_value_shift
#     target = target.astype(np.float32)
#     if initial:
#         transform = preprocessing.yxz_to_zyx_affine(nb.register.initial_transform[t, r, c])
#     else:
#         transform = preprocessing.yxz_to_zyx_affine(nb.register.transform[t, r, c])
#     target_transformed = affine_transform(target, transform, order=1)
#     # plot in napari
#     if filter:
#         anchor = sobel(anchor)
#         target = sobel(target)
#         target_transformed = sobel(target_transformed)
#
#     viewer = napari.Viewer()
#     viewer.add_image(anchor, name='Tile ' + str(t) + ', round ' + str(nb.basic_info.anchor_round) + ', channel ' +
#                                   str(nb.basic_info.anchor_channel), colormap='red', blending='additive')
#     viewer.add_image(target_transformed, name='Tile ' + str(t) + ', round ' + str(r) + ', channel ' + str(c) +
#                                               ' transformed', colormap='green', blending='additive')
#     viewer.add_image(target, name='Tile ' + str(t) + ', round ' + str(r) + ', channel ' + str(c), colormap='blue',
#                      blending='additive', opacity=0)
#     napari.run()


# TODO: This will need to be refactored. We will need to use flow instead of affine transforms
# def view_background_overlay(nb: Notebook, t: int, r: int, c: int):
#     """
#     Overlays tile t, round r, channel c with the preseq image for the same tile, and the same channel. The preseq image
#     is in red, and the seq image is in green. Both are registered.
#     Args:
#         nb: Notebook
#         t: tile
#         r: round
#         c: channel
#     """
#
#     if c in nb.basic_info.use_channels:
#         transform_preseq = preprocessing.yxz_to_zyx_affine(nb.register.transform[t, nb.basic_info.pre_seq_round, c])
#         transform_seq = preprocessing.yxz_to_zyx_affine(nb.register.transform[t, r, c])
#     elif c == 0:
#         transform_preseq = preprocessing.yxz_to_zyx_affine(nb.register.round_transform[t, nb.basic_info.pre_seq_round])
#         if r == nb.basic_info.anchor_round:
#             transform_seq = np.eye(4)
#         else:
#             transform_seq = preprocessing.yxz_to_zyx_affine(nb.register.round_transform[t, r])
#     seq = preprocessing.yxz_to_zyx(tiles_io.load_image(nb.file_names,nb.basic_info, nb.extract.file_type, t, r, c, apply_shift=False))
#     if not (r == nb.basic_info.anchor_round and c == nb.basic_info.dapi_channel):
#         seq = preprocessing.offset_pixels_by(seq, -nb.basic_info.tile_pixel_value_shift)
#     preseq = preprocessing.yxz_to_zyx(
#         tiles_io.load_image(
#             nb.file_names,nb.basic_info, nb.extract.file_type, t, nb.basic_info.pre_seq_round, c, apply_shift=False,
#         )
#     )
#     if not (nb.basic_info.pre_seq_round == nb.basic_info.anchor_round and c == nb.basic_info.dapi_channel):
#         preseq = preprocessing.offset_pixels_by(preseq, -nb.basic_info.tile_pixel_value_shift)
#
#     print('Starting Application of Seq Transform')
#     seq = affine_transform(seq, transform_seq, order=1)
#     print('Starting Application of Preseq Transform')
#     preseq = affine_transform(preseq, transform_preseq, order=1)
#     print('Finished Transformations')
#
#     viewer = napari.Viewer()
#     viewer.add_image(seq, name='seq', colormap='green', blending='additive')
#     viewer.add_image(preseq, name='preseq', colormap='red', blending='additive')
#     napari.run()

# TODO: This will need to be refactored as our method for computing background scale has changed
# def view_background_brightness_correction(nb: Notebook, t: int, r: int, c: int, percentile: int = 99,
#                                           sub_image_size: int = 500):
#     print(f"Computing background scale for tile {t}, round {r}, channel {c}")
#     mid_z = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
#     transform = nb.register.transform
#     r_pre = nb.basic_info.pre_seq_round
#
#     # Load in preseq
#     transform_pre = preprocessing.invert_affine(preprocessing.yxz_to_zyx_affine(transform[t, r_pre, c]))
#     z_scale_pre = transform_pre[0, 0]
#     z_shift_pre = transform_pre[0, 3]
#     mid_z_pre = int((mid_z - z_shift_pre) / z_scale_pre)
#     yxz = [None, None, mid_z_pre]
#     preseq = tiles_io.load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r_pre, c=c, yxz=yxz)
#     preseq = preseq.astype(np.int32)
#     preseq = preseq - nb.basic_info.tile_pixel_value_shift
#     preseq = preseq.astype(np.float32)
#     # we have to load in inverse transform to use scipy.ndimage.affine_transform
#     inv_transform_pre_yx = preprocessing.yxz_to_zyx_affine(transform[t, r_pre, c])[1:, 1:]
#     preseq = affine_transform(preseq, inv_transform_pre_yx)
#
#     # load in seq
#     transform_seq = preprocessing.invert_affine(preprocessing.yxz_to_zyx_affine(transform[t, r, c]))
#     z_scale_seq = transform_seq[0, 0]
#     z_shift_seq = transform_seq[0, 3]
#     mid_z_seq = int((mid_z - z_shift_seq) / z_scale_seq)
#     yxz = [None, None, mid_z_seq]
#     seq = tiles_io.load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r, c=c, yxz=yxz)
#     seq = seq.astype(np.int32)
#     seq = seq - nb.basic_info.tile_pixel_value_shift
#     seq = seq.astype(np.float32)
#     # we have to load in inverse transform to use scipy.ndimage.affine_transform
#     inv_transform_seq_yx = preprocessing.yxz_to_zyx_affine(transform[t, r, c])[1:, 1:]
#     seq = affine_transform(seq, inv_transform_seq_yx)
#
#     # compute bg scaling
#     bg_scale, sub_seq, sub_preseq = brightness_scale(preseq, seq, percentile, sub_image_size)
#     high_preseq = sub_preseq > np.percentile(sub_preseq, percentile)
#     positive = (sub_seq > 0) * (sub_preseq > 0)
#     mask = high_preseq * positive
#     diff = sub_seq - bg_scale * sub_preseq
#     ratio = sub_seq[mask] / sub_preseq[mask]
#     estimate_scales = np.percentile(ratio, [25, 75])
#     diff_low = sub_seq - estimate_scales[0] * sub_preseq
#     diff_high = sub_seq - estimate_scales[1] * sub_preseq
#
#     # View overlay and view regression
#     viewer = napari.Viewer()
#     viewer.add_image(sub_seq, name='seq', colormap='green', blending='additive')
#     viewer.add_image(sub_preseq, name='preseq', colormap='red', blending='additive')
#     viewer.add_image(diff, name='diff', colormap='gray', blending='translucent', visible=False)
#     viewer.add_image(diff_low, name='bg_scale = 25%', colormap='gray', blending='translucent', visible=False)
#     viewer.add_image(diff_high, name='bg_scale = 75%', colormap='gray', blending='translucent', visible=False)
#     viewer.add_image(mask, name='mask', colormap='blue', blending='additive', visible=False)
#
#     # View regression
#     plt.subplot(1, 2, 1)
#     bins = 25
#     plt.hist2d(sub_preseq[mask], sub_seq[mask], bins=[np.linspace(0, np.percentile(sub_preseq[mask], 90), bins),
#                                                       np.linspace(0, np.percentile(sub_seq[mask], 90), bins)])
#     x = np.linspace(0, np.percentile(sub_seq[mask], 90), 100)
#     y = bg_scale * x
#     plt.plot(x, y, 'r')
#     plt.plot(x, estimate_scales[0] * x, 'g')
#     plt.plot(x, estimate_scales[1] * x, 'g')
#     plt.xlabel('Preseq')
#     plt.ylabel('Seq')
#     plt.title('Regression of preseq vs seq. Scale = ' + str(np.round(bg_scale, 3)))
#
#     plt.subplot(1, 2, 2)
#     plt.hist(sub_seq[mask] / sub_preseq[mask], bins=100)
#     max_bin_val = np.max(np.histogram(sub_seq[mask] / sub_preseq[mask], bins=100)[0])
#     plt.vlines(bg_scale, 0, max_bin_val, colors='r')
#     plt.vlines(estimate_scales, 0, max_bin_val, colors='g')
#     plt.xlabel('Seq / Preseq')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of seq / preseq. Scale = ' + str(np.round(bg_scale, 3)))
#     plt.show()
#
#     napari.run()


def view_camera_correction(nb: Notebook):
    """
    Plots the camera correction for each camera against the anchor camera
    Args:
        nb: Notebook (must have register page and a path to fluorescent bead images)
    """
    # One transform for each camera
    viewer = napari.Viewer()
    fluorescent_bead_path = nb.file_names.fluorescent_bead_path
    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()

    if len(fluorescent_beads.shape) == 4:
        mid_z = fluorescent_beads.shape[0] // 2
        fluorescent_beads = fluorescent_beads[mid_z, :, :, :]
    # if fluorescent bead images are for all channels, just take one from each camera
    cam_channels = [0, 9, 18, 23]
    if len(fluorescent_beads) == 28:
        fluorescent_beads = fluorescent_beads[cam_channels]
    transform = nb.register.channel_transform[cam_channels][:, 1:, 1:]

    # Apply the transform to the fluorescent bead images
    fluorescent_beads_transformed = np.zeros(fluorescent_beads.shape)
    for c in range(4):
        fluorescent_beads_transformed[c] = affine_transform(fluorescent_beads[c], transform[c], order=3)

    # Add the images to napari
    colours = ["yellow", "red", "green", "blue"]
    for c in range(1, 4):
        viewer.add_image(
            fluorescent_beads[c],
            name="Camera " + str(cam_channels[c]),
            colormap=colours[c],
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            fluorescent_beads_transformed[c],
            name="Camera " + str(cam_channels[c]) + " transformed",
            colormap=colours[c],
            blending="additive",
            visible=True,
        )

    napari.run()

# TODO: Refactor this for icp correction instead of transform
def view_shifts_and_scales(nb: Notebook, t: int, bg_on: bool = False):
    """
    Plots the shifts and scales for tile t for each round and channel
    Args:
        nb: Notebook
        t: tile
        bg_on: boolean indicating whether to plot shifts and scales for preseq registration
    """
    use_rounds = [nb.basic_info.anchor_round + 1] * nb.basic_info.use_preseq * bg_on + nb.basic_info.use_rounds
    use_channels = nb.basic_info.use_channels
    transform_yxz = nb.register.transform[t][np.ix_(use_rounds, use_channels)]
    transform_zyx = np.zeros((len(use_rounds), len(use_channels), 3, 4))
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            transform_zyx[r, c] = preprocessing.yxz_to_zyx_affine(transform_yxz[r, c])
    scale = np.zeros((len(use_rounds), len(use_channels), 3))
    shift = np.zeros((len(use_rounds), len(use_channels), 3))
    # populate scale and shift
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            scale[r, c] = np.diag(transform_zyx[r, c, :3, :3])
            shift[r, c] = transform_zyx[r, c, :3, 3]

    # create plots
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    coord_label = ["Z", "Y", "X"]
    plot_label = ["Scale", "Shift"]
    image = [scale, shift]

    for i in range(2):
        for j in range(3):
            # plot the image
            ax[i, j].imshow(image[i][:, :, j].T)
            ax[i, j].set_xticks(np.arange(len(use_rounds)))
            ax[i, j].set_xticklabels(use_rounds)
            ax[i, j].set_yticks(np.arange(len(use_channels)))
            ax[i, j].set_yticklabels(use_channels)
            ax[i, j].set_title(plot_label[i] + " in " + coord_label[j])
            # for each subplot, assign a colour bar
            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(ax[i, j].get_images()[0], cax=cax)

            if j == 0:
                ax[i, j].set_ylabel("Channel")
            if i == 1:
                ax[i, j].set_xlabel("Round")
    plt.suptitle("Shifts and scales for tile " + str(t))
    plt.show()


class ViewSubvolReg:
    def __init__(self, nb: Notebook, t: int = None, r: int = None):
        """
        Class to view the subvolume registration for a given tile, round, channel.
        Args:
            nb: Notebook
            t: tile
            r: round
        """
        self.nb, self.t, self.r = nb, t, r
        if self.t is None:
            self.t = nb.basic_info.use_tiles[0]
        if self.r is None:
            self.r = nb.basic_info.use_rounds[0]
        # load in shifts
        self.shift = nb.register_debug.round_shift[self.t, self.r]
        self.shift_corr = nb.register_debug.round_shift_corr[self.t, self.r]
        self.position = nb.register_debug.position
        shift_prediction_matrix = huber_regression(self.shift, self.position)
        self.predicted_shift = np.pad(self.position, ((0, 0), (0, 1)), constant_values=1) @ shift_prediction_matrix.T
        self.shift_residual = np.linalg.norm(self.shift - self.predicted_shift, axis=-1)
        # load in subvolumes
        self.subvol_base, self.subvol_target = None, None
        self.subvol_z, self.subvol_y, self.subvol_x = None, None, None
        self.box_z, self.box_y, self.box_x = None, None, None
        self.load_subvols()
        # reshape shifts
        self.shift = self.shift.reshape((self.subvol_z, self.subvol_y, self.subvol_x, 3))
        self.shift_corr = self.shift_corr.reshape((self.subvol_z, self.subvol_y, self.subvol_x))
        self.position = self.position.reshape((self.subvol_z, self.subvol_y, self.subvol_x, 3)).astype(int)
        self.predicted_shift = self.predicted_shift.reshape((self.subvol_z, self.subvol_y, self.subvol_x, 3))
        self.shift_residual = self.shift_residual.reshape((self.subvol_z, self.subvol_y, self.subvol_x))
        self.transform = nb.register_debug.round_transform_raw[self.t, self.r]
        # create viewer
        self.viewer = napari.Viewer()
        napari.run()

    def load_subvols(self):
        """
        Load in subvolumes for the given tile and round
        """
        # load in images
        config = self.nb.get_config()['register']
        round_registration_channel = 0
        anchor_image = preprocessing.yxz_to_zyx(tiles_io.load_image(self.nb.file_names, self.nb.basic_info, self.nb.extract.file_type,
                                                      self.t, self.nb.basic_info.anchor_round,
                                                      round_registration_channel, apply_shift=False))
        round_image = preprocessing.yxz_to_zyx(tiles_io.load_image(self.nb.file_names, self.nb.basic_info, self.nb.extract.file_type,
                                                     self.t, self.r, round_registration_channel, apply_shift=False))

        # split images into subvolumes
        z_subvols, y_subvols, x_subvols = config["subvols"]
        z_box, y_box, x_box = config["box_size"]
        subvol_base, _ = preprocessing.split_3d_image(
            image=anchor_image,
            z_subvolumes=z_subvols,
            y_subvolumes=y_subvols,
            x_subvolumes=x_subvols,
            z_box=z_box,
            y_box=y_box,
            x_box=x_box,
        )
        subvol_target, _ = preprocessing.split_3d_image(
            image=round_image,
            z_subvolumes=z_subvols,
            y_subvolumes=y_subvols,
            x_subvolumes=x_subvols,
            z_box=z_box,
            y_box=y_box,
            x_box=x_box,
        )
        self.subvol_z, self.subvol_y, self.subvol_x = int(z_subvols), int(y_subvols), int(x_subvols)
        self.box_z, self.box_y, self.box_x = int(z_box), int(y_box), int(x_box)
        self.subvol_base, self.subvol_target = subvol_base, subvol_target

    def view_subvol_cross_corr(self, z: int = 0, y: int = 0, x: int = 0, grid_view: bool = False):
        """
        View the cross correlation for a single subvolume for the given tile and round
        Args:
            z: z index of subvolume
            y: y index of subvolume
            x: x index of subvolume
            grid_view: whether to view the images in a 2D grid, or as a 3D stack
        """
        # check if there are any layers in the viewer, if so, remove them
        if len(self.viewer.layers) > 0:
            for i in range(len(self.viewer.layers)):
                self.viewer.layers.pop()
        z_start, z_end = int(max(0, z - 1)), int(min(self.subvol_z, z + 1) + 1)
        merged_subvol_target = preprocessing.merge_subvols(
            position=self.position[z_start:z_end, y, x].copy(), subvol=self.subvol_target[z_start:z_end, y, x]
        )
        merged_subvol_target_windowed = preprocessing.window_image(merged_subvol_target)
        merged_subvol_base = np.zeros_like(merged_subvol_target)
        merged_subvol_base_windowed = np.zeros_like(merged_subvol_target)
        merged_subvol_min_z = self.position[z_start, y, x][0]
        current_box_min_z = self.position[z, y, x][0]
        merged_subvol_start_z = current_box_min_z - merged_subvol_min_z
        merged_subvol_base[merged_subvol_start_z : merged_subvol_start_z + self.box_z] = self.subvol_base[z, y, x]
        merged_subvol_base_windowed[merged_subvol_start_z : merged_subvol_start_z + self.box_z] = (
            preprocessing.window_image(self.subvol_base[z, y, x])
        )

        # compute cross correlation
        im_centre = np.array(merged_subvol_base_windowed.shape) // 2
        f_hat = fft.fftn(merged_subvol_base_windowed)
        g_hat = fft.fftn(merged_subvol_target_windowed)
        phase_cross = f_hat * np.conj(g_hat) / (np.abs(f_hat) * np.abs(g_hat))
        phase_cross_ifft = fft.fftshift(np.abs(fft.ifftn(phase_cross)))
        phase_cross_shift = -(np.unravel_index(np.argmax(phase_cross_ifft), phase_cross_ifft.shape) - im_centre)

        # add images
        y_size, x_size = merged_subvol_base.shape[1:]
        if not grid_view:
            self.viewer.add_image(phase_cross_ifft, name=f"Phase cross correlation. z = {z}, y = {y}, x = {x}")
            self.viewer.add_points(
                [-phase_cross_shift + im_centre],
                name="Phase cross correlation shift",
                size=5,
                face_color="blue",
                symbol="cross",
            )
            # add overlays below this
            translation_offset = np.array([0, 1.1 * y_size, 0])
            self.viewer.add_image(
                merged_subvol_target,
                name=f"Target. z = {z}, y = {y}, x = {x}",
                colormap="green",
                blending="additive",
                translate=translation_offset,
            )
            self.viewer.add_image(
                merged_subvol_base,
                name=f"Base. Shift = {phase_cross_shift}",
                colormap="red",
                blending="additive",
                translate=translation_offset + phase_cross_shift,
            )
            # add predicted shift
            translation_offset = np.array([0, 1.1 * y_size, 1.1 * x_size])
            self.viewer.add_image(
                merged_subvol_target,
                name=f"Target. z = {z}, y = {y}, x = {x}",
                colormap="green",
                blending="additive",
                translate=translation_offset,
            )
            self.viewer.add_image(
                merged_subvol_base,
                name=f"Base. Predicted Shift = " f"{np.rint(self.predicted_shift[z, y, x])}",
                colormap="red",
                blending="additive",
                translate=translation_offset + self.predicted_shift[z, y, x],
            )
        else:
            # generate transformed image
            new_origin = np.array([merged_subvol_min_z, self.position[z, y, x, 1], self.position[z, y, x, 2]])
            transform = (self.transform).copy()
            # need to adjust shift as we are changing the origin
            transform[:, 3] += (transform[:3, :3] - np.eye(3)) @ new_origin
            transform = preprocessing.invert_affine(transform)
            merged_subvol_base_transformed = affine_transform(merged_subvol_base, transform, order=0)

            # initialise grid
            phase_cross_shift_yx = phase_cross_shift[1:]
            predicted_shift_yx = self.predicted_shift[z, y, x, 1:]
            nz = merged_subvol_target.shape[0]
            features_z = {"z": np.arange(nz)}
            text_z = {"string": "Z: {z}", "size": 8, "color": "white"}
            for i in range(nz):
                # 1. Plot cross correlation
                translation_offset = np.array([0, 1.1 * x_size * i])
                self.viewer.add_image(
                    phase_cross_ifft[i],
                    name=f"Phase cross correlation. z = {z}, y = {y}, x = {x}",
                    translate=translation_offset,
                )
                # 2. Plot base and target, with base shifted by phase_cross_shift
                translation_offset = np.array([1.1 * y_size, 1.1 * x_size * i])
                self.viewer.add_image(
                    merged_subvol_target[i],
                    name=f"Target. z = {z}, y = {y}, x = {x}",
                    colormap="green",
                    blending="additive",
                    translate=translation_offset,
                )
                base_i = (i - np.rint(self.shift[z, y, x, 0])).astype(int)
                if (base_i >= 0) and (base_i < nz):
                    self.viewer.add_image(
                        merged_subvol_base[base_i],
                        name=f"Base. Shift = {phase_cross_shift}",
                        colormap="red",
                        blending="additive",
                        translate=translation_offset + phase_cross_shift_yx,
                    )
                # 3. Plot base and target, with base shifted by predicted_shift
                translation_offset = np.array([2.2 * y_size, 1.1 * x_size * i])
                self.viewer.add_image(
                    merged_subvol_target[i],
                    name=f"Target. z = {z}, y = {y}, x = {x}",
                    colormap="green",
                    blending="additive",
                    translate=translation_offset,
                )
                base_i = (i - np.rint(self.predicted_shift[z, y, x, 0])).astype(int)
                if (base_i >= 0) and (base_i < nz):
                    self.viewer.add_image(
                        merged_subvol_base[base_i],
                        name=f"Base. Predicted Shift = " f"{np.rint(self.predicted_shift[z, y, x])}",
                        colormap="red",
                        blending="additive",
                        translate=translation_offset + predicted_shift_yx,
                    )
                # 4. Plot affine transformed image
                translation_offset = np.array([3.3 * y_size, 1.1 * x_size * i])
                self.viewer.add_image(
                    merged_subvol_target[i],
                    name=f"Target. z = {z}, y = {y}, x = {x}",
                    colormap="green",
                    blending="additive",
                    translate=translation_offset,
                )
                self.viewer.add_image(
                    merged_subvol_base_transformed[i],
                    name=f"Base. Affine transformed",
                    colormap="red",
                    blending="additive",
                    translate=translation_offset,
                )

            # plot z plane numbers above each z plane
            z_label_coords = [np.array([-20, 1.1 * x_size * i + x_size // 2]) for i in range(nz)]
            self.viewer.add_points(z_label_coords, features=features_z, text=text_z, size=0)
            self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)
            self.viewer.window.qt_viewer.dockLayerList.setVisible(False)

        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        napari.run()


class ButtonSubvolWindow(QMainWindow):
    def __init__(self, nz: int, ny: int, nx: int, active_button: list = [0, 0, 0]):
        super().__init__()
        # Loop through subvolumes and create a button for each
        for z, y, x in np.ndindex(nz, ny, nx):
            # Create a button for each subvol
            button = QPushButton(str([z, y, x]), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            button.setGeometry(500 * z + y * 70, x * 40, 50, 28)
            # set active button as checked
            if active_button == [z, y, x]:
                button.setChecked(True)
                self.subvol = [z, y, x]
            # Set button color = grey when hovering over
            # set colour of tiles in use to blue amd not in use to red
            button.setStyleSheet(
                "QPushButton"
                "{"
                "background-color : rgb(135, 206, 250);"
                "}"
                "QPushButton::hover"
                "{"
                "background-color : lightgrey;"
                "}"
                "QPushButton::pressed"
                "{"
                "background-color : white;"
                "}"
            )
            # Finally add this button as an attribute to self
            self.__setattr__(str([z, y, x]), button)
