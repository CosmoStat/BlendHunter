import os
import numpy as np
import cv2

class denoise_test_images(object):

    def __init__(self, test_images, output_path):

        self.images = test_images
        self.path = output_path
        self._make_output_dirs()

    def _make_output_dirs(self):
        """ Make Output Directories

        This method creates the directories where the samples will be stored.

        """

        images_path = '{}/denoised_test_images'.format(self.path)


        if os.path.isdir(images_path):
            raise FileExistsError('{} already exists. Please remove this '
                                  'directory or choose a new path.'
                                  ''.format(images_path))

        os.mkdir(images_path)


    def call_mr_filter(data, opt='', path='./',
                      remove_files=True):  # pragma: no cover
    r"""Call mr_transform
    This method calls the iSAP module mr_filter
    Parameters
    ----------
    data : np.ndarray
        Input data, 2D array
    opt : list or str, optional
        Options to be passed to mr_transform
    path : str, optional
        Path for output files (default is './')
    remove_files : bool, optional
        Option to remove output files (default is 'True')
    Returns
    -------
    np.ndarray results of mr_filter
    Raises
    ------
    ValueError
        If the input data is not a 2D numpy array
    Examples
    """

    if not import_astropy:
        raise ImportError('Astropy package not found.')

    if (not isinstance(data, np.ndarray)) or (data.ndim != 2):
        raise ValueError('Input data must be a 2D numpy array.')

    executable = 'mr_filter'

    # Make sure mr_filter is installed.
    is_executable(executable)

    # Create a unique string using the current date and time.
    unique_string = (datetime.now().strftime('%Y.%m.%d_%H.%M.%S') +
                     str(getrandbits(128)))

    # Set the ouput file names.
    file_name = path + 'mr_temp_' + unique_string
    file_fits = file_name + '.fits'
    file_mr = file_name + '.mr'

    # Write the input data to a fits file.
    fits.writeto(file_fits, data)

    if isinstance(opt, str):
        opt = opt.split()

    # Prepare command and execute it
    command_line = ' '.join([executable] + opt + [file_fits, file_mr])
    stdout, stderr = execute(command_line)

    # Check for errors
    if any(word in stdout for word in ('bad', 'Error')):

        remove(file_fits)
        raise RuntimeError('{} raised following exception: "{}"'
                           ''.format(executable, stdout.rstrip('\n')))

    # Retrieve wavelet transformed data.
    result = fits.getdata(file_mr)

    # Remove the temporary files.
    if remove_files:
        remove(file_fits)
        remove(file_mr)

    # Return the mr_transform results.
    return result


    @staticmethod
    def _rescale(array):
        """ Rescale

        Rescale input image to RGB.

        Parameters
        ----------
        array : np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Rescaled array

        """

        array = np.abs(array)

        return np.array(array * 255 / np.max(array)).astype(int)


    @staticmethod
    def _pad(array, padding):
        """ Pad

        Pad array with specified padding.

        Parameters
        ----------
        array : np.ndarray
            Input array
        padding : np.ndarray
            Padding amount

        Returns
        -------
        np.ndarray
            Padded array

        """

        x, y = padding + padding % 2

        return np.pad(array, ((x, x), (y, y)), 'constant')


    def _write_images(self, images, path):
        """ Write Images

        Write images to jpeg files.

        Parameters
        ----------
        images : np.ndarray
            Array of images
        path : str
            Path where images should be written

        """

        min_shape = np.array([48, 48])

        for image in images:

            image = self._rescale(image)

            shape_diff = (min_shape - np.array(image.shape))[:2]

            if np.sum(shape_diff) > 0:
                image = self._pad(image, shape_diff)

            cv2.imwrite('{}/image_{}.png'.format(path, self._image_num), image)
            self._image_num += 1
