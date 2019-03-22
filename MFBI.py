import numpy as np
import rasterio
from skimage.exposure import rescale_intensity
from sklearn.decomposition import PCA
from tqdm import tqdm



def grayscale_raster_creation(input_MS_file, output_filename):
    """ 
    This function creates a grayscale brightness image from an input image to be used for multi - scale - texture index calculation. 
    For every pixel in the input image, the intensity values from the red, green, blue channels are first obtained, and the maximum of 
    these values are then assigned as the pixel's intensity value, which would give the grayscale brightness image as mentioned earlier, 
    as per standard practice in the remote sensing academia and industry. It is assumed that the first three channels of the input 
    image correspond to the red, green and blue channels, irrespective of order.
    
    Inputs:
    - input_MS_file: File path of the input image that needs to be converted to grayscale brightness image
    - output_filename: File path of the grayscale brightness image that is to be written to file
    
    Outputs:
    - gray: Numpy array of grayscale brightness image of corresponding multi - channel input image
    
    """
    
    with rasterio.open(input_MS_file) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
    gray = np.zeros((int(img.shape[0]), int(img.shape[1]), 1))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray[i, j, 0] = max(img[i, j, 0], img[i, j, 1], img[i, j, 2])
            
    gray = np.transpose(gray, [2, 0, 1]).astype('float32')
    
    metadata['count'] = 1
    metadata['dtype'] = 'float32'
    
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(gray)
    
    return gray



def MFBI_generation(input_gray_filename, output_filename, min_scale, max_scale, step_size):
    """ 
    This function is used to create the Multi - Scale Filtering Building Index (MFBI) defined in the paper 
    'A Multi - Scale Filtering Building Index for Building Extraction in Very High - Resolution Satellite Imagery' by 
    Bi Q., Qin K., Zhang H., Zhang Y., Li Z., Xu K. (2019). 
    
    Inputs:
    - input_gray_filename: String or path of input grayscale image to be used
    - output_filename: String or path of MFBI feature image to be written
    - min_scale: Minimum sliding window size to be applied across the original grayscale image (must be an odd number)
    - max_scale: Maximum sliding window size to be applied across the original grayscale image (must be an odd number)
    - step_size: Spatial increment of sliding window size (must be an even number)
    
    Output:
    - MFBI: Numpy array which represents the MFBI feature image
    
    """

    if (step_size % 2 != 0):
        raise ValueError('Please input an even number for step_size.')
    
    if (min_scale % 2 == 0):
        raise ValueError('Please input an odd number for min_scale.')
    
    if (max_scale % 2 == 0):
        raise ValueError('Please input an odd number for max_scale.')
    
    with rasterio.open(input_gray_filename) as f:
        metadata = f.profile
        gray_img = f.read(1)
    
    max_buffer = int((max_scale - 1) / 2)
    gray_img_padded = np.zeros(((gray_img.shape[0] + 2 * max_buffer), (gray_img.shape[1] + 2 * max_buffer)))
    gray_img_padded[max_buffer : (max_buffer + gray_img.shape[0]), max_buffer : (max_buffer + gray_img.shape[1])] = gray_img
    
    FP_img = np.zeros((gray_img.shape[0], gray_img.shape[1], int((max_scale - min_scale) / step_size) + 1))
    
    for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
        buffer = int((scale - 1) / 2)
        for i in range(max_buffer, gray_img_padded.shape[0] - max_buffer):            
            for j in range(max_buffer, gray_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                array = gray_img_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                FP_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = np.mean(array)
                
    DFP_list = []
    
    for band in range(int((max_scale - min_scale) / step_size)):
        DFP = np.abs(FP_img[:, :, band + 1] - FP_img[:, :, band])
        DFP_list.append(np.expand_dims(DFP, axis = 2))
        
    DFP_img = np.concatenate(DFP_list, axis = 2)
    
    MFBI = np.expand_dims(np.mean(DFP_img, axis = 2, dtype = metadata['dtype']), axis = 2)
    MFBI = rescale_intensity(MFBI, out_range = (0, 1)).astype(metadata['dtype'])
    
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(np.transpose(MFBI, [2, 0, 1]))
        
    return MFBI



def MMFBI_v1_creation(input_MS_filename, output_filename, min_scale, max_scale, step_size):
    """ 
    This function is used to create the Multi - Channel Multi - Scale Filtering Building Index (MMFBI) (Scenario 1)
    defined in the paper 
    'A Multi - Scale Filtering Building Index for Building Extraction in Very High - Resolution Satellite Imagery' by 
    Bi Q., Qin K., Zhang H., Zhang Y., Li Z., Xu K. (2019). 
    
    Inputs:
    - input_MS_filename: String or path of input multispectral image to be used
    - output_filename: String or path of MFBI feature image to be written
    - min_scale: Minimum sliding window size to be applied across the original grayscale image (must be an odd number)
    - max_scale: Maximum sliding window size to be applied across the original grayscale image (must be an odd number)
    - step_size: Spatial increment of sliding window size (must be an even number)
    
    Output:
    - MMFBI_v1: Numpy array which represents the MMFBI feature image (Scenario 1)
    
    """
    
    if (step_size % 2 != 0):
        raise ValueError('Please input an even number for step_size.')
    
    if (min_scale % 2 == 0):
        raise ValueError('Please input an odd number for min_scale.')
    
    if (max_scale % 2 == 0):
        raise ValueError('Please input an odd number for max_scale.')
    
    with rasterio.open(input_MS_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    band_list = []
    
    for band in range(img.shape[2]):
        band_list.append(np.reshape(img[:, :, band], (-1, 1)))
    band_array = np.concatenate(band_list, axis = 1)
    
    pca_components = PCA().fit_transform(band_array)
    pc1 = np.reshape(pca_components[:, 0], (img.shape[0], img.shape[1]))
    
    max_buffer = int((max_scale - 1) / 2)
    pc1_img_padded = np.zeros(((img.shape[0] + 2 * max_buffer), (img.shape[1] + 2 * max_buffer)))
    pc1_img_padded[max_buffer : (max_buffer + img.shape[0]), max_buffer : (max_buffer + img.shape[1])] = pc1
    
    FP_img = np.zeros((img.shape[0], img.shape[1], int((max_scale - min_scale) / step_size) + 1))
    
    for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
        buffer = int((scale - 1) / 2)
        for i in range(max_buffer, pc1_img_padded.shape[0] - max_buffer):            
            for j in range(max_buffer, pc1_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                array = pc1_img_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                FP_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = np.mean(array)
                
    DFP_list = []
    
    for band in range(int((max_scale - min_scale) / step_size)):
        DFP = np.abs(FP_img[:, :, band + 1] - FP_img[:, :, band])
        DFP_list.append(np.expand_dims(DFP, axis = 2))
        
    DFP_img = np.concatenate(DFP_list, axis = 2)
    
    metadata['dtype'] = 'float32'
    
    MMFBI_v1 = np.expand_dims(np.mean(DFP_img, axis = 2, dtype = metadata['dtype']), axis = 2)
    MMFBI_v1 = rescale_intensity(MMFBI_v1, out_range = (0, 1)).astype(metadata['dtype'])
    
    metadata['count'] = 1
    
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(np.transpose(MMFBI_v1, [2, 0, 1]))
        
    return MMFBI_v1



def MMFBI_v2_creation(input_MS_filename, output_filename, min_scale, max_scale, step_size):
    """ 
    This function is used to create the Multi - Channel Multi - Scale Filtering Building Index (MMFBI) (Scenario 2)
    defined in the paper 
    'A Multi - Scale Filtering Building Index for Building Extraction in Very High - Resolution Satellite Imagery' by 
    Bi Q., Qin K., Zhang H., Zhang Y., Li Z., Xu K. (2019). 
    
    Inputs:
    - input_MS_filename: String or path of input multispectral image to be used
    - output_filename: String or path of MFBI feature image to be written
    - min_scale: Minimum sliding window size to be applied across the original grayscale image (must be an odd number)
    - max_scale: Maximum sliding window size to be applied across the original grayscale image (must be an odd number)
    - step_size: Spatial increment of sliding window size (must be an even number)
    
    Output:
    - MMFBI_v2: Numpy array which represents the MMFBI feature image (Scenario 2)
    
    """
    
    if (step_size % 2 != 0):
        raise ValueError('Please input an even number for step_size.')
    
    if (min_scale % 2 == 0):
        raise ValueError('Please input an odd number for min_scale.')
    
    if (max_scale % 2 == 0):
        raise ValueError('Please input an odd number for max_scale.')
    
    with rasterio.open(input_MS_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    metadata['dtype'] = 'float32'
    max_buffer = int((max_scale - 1) / 2)
        
    individual_MFBI_list = []
    
    for band in range(img.shape[2]):
        gray = img[:, :, band]
        gray_img_padded = np.zeros(((img.shape[0] + 2 * max_buffer), (img.shape[1] + 2 * max_buffer)))
        gray_img_padded[max_buffer : (max_buffer + img.shape[0]), max_buffer : (max_buffer + img.shape[1])] = gray
        
        FP_img = np.zeros((img.shape[0], img.shape[1], int((max_scale - min_scale) / step_size) + 1))
        
        for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
            buffer = int((scale - 1) / 2)
            for i in range(max_buffer, gray_img_padded.shape[0] - max_buffer):            
                for j in range(max_buffer, gray_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                    array = gray_img_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                    FP_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = np.mean(array)
                    
        DFP_list = []
    
        for scale in range(int((max_scale - min_scale) / step_size)):
            DFP = np.abs(FP_img[:, :, scale + 1] - FP_img[:, :, scale])
            DFP_list.append(np.expand_dims(DFP, axis = 2))
            
        DFP_img = np.concatenate(DFP_list, axis = 2)
       
        MFBI = np.expand_dims(np.mean(DFP_img, axis = 2, dtype = metadata['dtype']), axis = 2)
        individual_MFBI_list.append(MFBI)
        
        if band != (img.shape[2] - 1):
            print('MFBI for band {} has been calculated.'.format(int(band + 1)))
        else:
            print('MFBI for final band has been calculated.')
        
    MFBI_img = np.concatenate(individual_MFBI_list, axis = 2)
    
    band_list = []
    
    for band in range(img.shape[2]):
        band_list.append(np.reshape(MFBI_img[:, :, band], (-1, 1)))
    band_array = np.concatenate(band_list, axis = 1)
    
    pca_components = PCA().fit_transform(band_array)
    pc1 = np.reshape(pca_components[:, 0], (img.shape[0], img.shape[1]))
    
    pc1_img_padded = np.zeros(((img.shape[0] + 2 * max_buffer), (img.shape[1] + 2 * max_buffer)))
    pc1_img_padded[max_buffer : (max_buffer + img.shape[0]), max_buffer : (max_buffer + img.shape[1])] = pc1
    
    FP_img = np.zeros((img.shape[0], img.shape[1], int((max_scale - min_scale) / step_size) + 1))
    
    for scale in tqdm(range(min_scale, max_scale + 1, step_size), mininterval = 300):
        buffer = int((scale - 1) / 2)
        for i in range(max_buffer, pc1_img_padded.shape[0] - max_buffer):            
            for j in range(max_buffer, pc1_img_padded.shape[1] - max_buffer) :                                                                                                                                   
                array = pc1_img_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)]
                FP_img[i - max_buffer, j - max_buffer, int((scale - min_scale) / step_size)] = np.mean(array)
                
    DFP_list = []
    
    for band in range(int((max_scale - min_scale) / step_size)):
        DFP = np.abs(FP_img[:, :, band + 1] - FP_img[:, :, band])
        DFP_list.append(np.expand_dims(DFP, axis = 2))
        
    DFP_img = np.concatenate(DFP_list, axis = 2)
    
    MMFBI_v2 = np.expand_dims(np.mean(DFP_img, axis = 2, dtype = metadata['dtype']), axis = 2)
    MMFBI_v2 = rescale_intensity(MMFBI_v2, out_range = (0, 1)).astype(metadata['dtype'])
    
    metadata['count'] = 1
    
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(np.transpose(MMFBI_v2, [2, 0, 1]))
        
    return MMFBI_v2