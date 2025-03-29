import pickle
import numpy as np

def load(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def save(save_path , obj):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

file_names = ['lc_classnum_Ia.pickle', # Shortened version of actual filenames
 'lc_classnum_Ia-91bg.pickle', 
 'lc_classnum_Iax.pickle', 
'lc_classnum_Ib.pickle', 
 'lc_classnum_Ic.pickle',  
 'lc_classnum_Ic-BL.pickle',  
 'lc_classnum_II.pickle', 
 'lc_classnum_IIn.pickle', 
 'lc_classnum_IIb.pickle', 
 'lc_classnum_TDE.pickle', 
 'lc_classnum_SLSN-I.pickle', 
 'lc_classnum_AGN_old.pickle',
 'lc_classnum_CART_old.pickle',
 'lc_classnum_Kilonova.pickle',
'lc_classnum_PISN_old.pickle',
'lc_classnum_ILOT_old.pickle',
'lc_classnum_uLens-BSR.pickle']

classes = ['SNIa', 'SNIa-91bg', 'SNIax', 'SNIb', 'SNIc', 'SNIc-BL', 'SNII', 'SNIIn', 'SNIIb', 'TDE', 'SLSN-I', 'AGN', 'CaRT', 'KNe', 'PISN', 'ILOT', 'uLens-BSR'] # In order with file names

file_to_class = dict(zip(file_names, classes)) # Convert from file value to the classname


class_to_file = {v: k for k, v in file_to_class.items()} # Convert from class to filename
class_to_file['SNIa-x'] = class_to_file['SNIax']
class_to_file['SNIa-norm'] = class_to_file['SNIa']

anom_classes = classes[-5:]
non_anom_classes = classes[:-5]

colors = ['r', 'g', 'y', 'b', 'purple', 'orange', 'gray', 'k', 'm', 'c', 'brown', 'olive'] # by class

ordered_class_names = classes

def create_ordered_class_names(enc):
    dummy = enc.transform(np.array(non_anom_classes).reshape(-1, 1))
    ordered_class_names = [-1 for i in range(len(non_anom_classes))]
    for ind, i in enumerate(dummy.todense()):
        ordered_class_names[np.argmax(i)] = non_anom_classes[ind]