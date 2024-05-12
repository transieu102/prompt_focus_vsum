import h5py
import cv2
# import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import os
# TVSUM name & video_id
file_name_to_h5 = {'0tmA_C6XwfM.mp4': 'video_13', 'EE-bNr36nyA.mp4': 'video_38', 'JgHubY5Vw3Y.mp4': 'video_44', 'VuWGsYPqAX8.mp4': 'video_31', '37rzWOQsNIw.mp4': 'video_19', 'eQu1rNs0an0.mp4': 'video_43', 'JKpqYvAdIsw.mp4': 'video_32', '3eYKfiOEJNs.mp4': 'video_14', '-esJrBWj2d8.mp4': 'video_50', 'kLxoNp-UchI.mp4': 'video_48', 'WxtbjNsCQ8A.mp4': 'video_36', '4wU_LUjG5Ic.mp4': 'video_30', 'EYqVtI9YWJA.mp4': 'video_42', 'LRw_obCPUt0.mp4': 'video_20', 'XkqCExn6_Us.mp4': 'video_23', '91IHQYk1IQM.mp4': 'video_26', 'fWutDQy1nnY.mp4': 'video_29', 'NyBmCxDoHJU.mp4': 'video_47', 'xmEERLqJ2kU.mp4': 'video_33', '98MoyGZKHXc.mp4': 'video_2', 'GsAD1KT1xo8.mp4': 'video_24', 'oDXZc0tZe04.mp4': 'video_40', '_xMr-HKMfVA.mp4': 'video_35', 'akI8YFjEmUw.mp4': 'video_10', 'gzDbaEs1Rlg.mp4': 'video_4', 'PJrm840pAUI.mp4': 'video_25', 'xwqBXPGE9pQ.mp4': 'video_9', 'AwmHb44_ouw.mp4': 'video_1', 'Hl-__g2gn_A.mp4': 'video_17', 'qqR6AEXwxoQ.mp4': 'video_41', 'xxdtq8mxegs.mp4': 'video_15', 'b626MiF1ew4.mp4': 'video_22', 'HT5vyqe0Xaw.mp4': 'video_6', 'RBCABdttQmI.mp4': 'video_27', 'XzYM3PfTM4w.mp4': 'video_5', 'Bhxk-O1Y7Ho.mp4': 'video_12', 'i3wAGJaaktw.mp4': 'video_11', 'Yi4Ij2NM7U4.mp4': 'video_18', 'byxOvuiIJV0.mp4': 'video_34', 'iVt07TCkFM0.mp4': 'video_45', 'sTEELN-vY30.mp4': 'video_7', 'z_6gVvQb2d0.mp4': 'video_28', 'cjibtmSLxQ4.mp4': 'video_21', 'J0nA4VgnoCo.mp4': 'video_3', 'uGu_10sucQo.mp4': 'video_37', 'E11zDS9XGzg.mp4': 'video_46', 'jcoYJXDG9sw.mp4': 'video_49', 'vdmoEJ5YbrQ.mp4': 'video_8', 'WG0MBPpPC6I.mp4': 'video_16', 'Se3oxnaPsz0.mp4': 'video_39'}
h5_to_file_name = {'video_13': '0tmA_C6XwfM.mp4', 'video_38': 'EE-bNr36nyA.mp4', 'video_44': 'JgHubY5Vw3Y.mp4', 'video_31': 'VuWGsYPqAX8.mp4', 'video_19': '37rzWOQsNIw.mp4', 'video_43': 'eQu1rNs0an0.mp4', 'video_32': 'JKpqYvAdIsw.mp4', 'video_14': '3eYKfiOEJNs.mp4', 'video_50': '-esJrBWj2d8.mp4', 'video_48': 'kLxoNp-UchI.mp4', 'video_36': 'WxtbjNsCQ8A.mp4', 'video_30': '4wU_LUjG5Ic.mp4', 'video_42': 'EYqVtI9YWJA.mp4', 'video_20': 'LRw_obCPUt0.mp4', 'video_23': 'XkqCExn6_Us.mp4', 'video_26': '91IHQYk1IQM.mp4', 'video_29': 'fWutDQy1nnY.mp4', 'video_47': 'NyBmCxDoHJU.mp4', 'video_33': 'xmEERLqJ2kU.mp4', 'video_2': '98MoyGZKHXc.mp4', 'video_24': 'GsAD1KT1xo8.mp4', 'video_40': 'oDXZc0tZe04.mp4', 'video_35': '_xMr-HKMfVA.mp4', 'video_10': 'akI8YFjEmUw.mp4', 'video_4': 'gzDbaEs1Rlg.mp4', 'video_25': 'PJrm840pAUI.mp4', 'video_9': 'xwqBXPGE9pQ.mp4', 'video_1': 'AwmHb44_ouw.mp4', 'video_17': 'Hl-__g2gn_A.mp4', 'video_41': 'qqR6AEXwxoQ.mp4', 'video_15': 'xxdtq8mxegs.mp4', 'video_22': 'b626MiF1ew4.mp4', 'video_6': 'HT5vyqe0Xaw.mp4', 'video_27': 'RBCABdttQmI.mp4', 'video_5': 'XzYM3PfTM4w.mp4', 'video_12': 'Bhxk-O1Y7Ho.mp4', 'video_11': 'i3wAGJaaktw.mp4', 'video_18': 'Yi4Ij2NM7U4.mp4', 'video_34': 'byxOvuiIJV0.mp4', 'video_45': 'iVt07TCkFM0.mp4', 'video_7': 'sTEELN-vY30.mp4', 'video_28': 'z_6gVvQb2d0.mp4', 'video_21': 'cjibtmSLxQ4.mp4', 'video_3': 'J0nA4VgnoCo.mp4', 'video_37': 'uGu_10sucQo.mp4', 'video_46': 'E11zDS9XGzg.mp4', 'video_49': 'jcoYJXDG9sw.mp4', 'video_8': 'vdmoEJ5YbrQ.mp4', 'video_16': 'WG0MBPpPC6I.mp4', 'video_39': 'Se3oxnaPsz0.mp4'}


def create_h5_file(video_names : list, data : dict, h5_file_name : str) -> None:
    """
    create h5 file
    Args:
    video_names : list
        list of video names
    data : dict 
        dict of data
    h5_file_name : str
        h5 file name or save path
    """
    f = h5py.File(h5_file_name, 'w')
    for name in video_names:
        for key in data[name].keys():
            f.create_dataset(name + '/' + key, data=data[name][key])
    f.close()

def read_h5_file(h5_file_name : str) -> dict:
    """
    read h5 file
    Args:
    h5_file_name : str
        h5 file name or save path
    """
    f = h5py.File(h5_file_name, 'r')
    data = {}
    for name in f.keys():
        data[name] = {}
        for key in f[name].keys():
            print(key)
            data[name][key] = f[name + '/' + key][()]
    return data


def extract_features(image_cv2 : np.ndarray, extractor : object, preprocess : object, device = torch.device('cpu')) -> np.ndarray:
    """
    extract features
    Args:
    image_cv2 : np.ndarray
        image
    extractor : object
        feature extractor
    device = torch.device('cpu')
        device
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load and preprocess the image
    image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    # Extract features using the model
    with torch.no_grad():
        features = extractor(image)

    # Return the extracted features
    return features.squeeze().cpu().numpy()
