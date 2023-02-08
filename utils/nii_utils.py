import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np

def load_nii(filename):
    print('Loading file: ' + filename)
    nii_slices = nib.load(filename).get_fdata()
    return nii_slices


def show_nii_slices(nii_slices, f_slicenum = 0, l_slicenum = 0):
    if l_slicenum == 0:
        l_slicenum = np.shape(nii_slices)[2]
    for i in range(f_slicenum, l_slicenum):
        plt.subplot((l_slicenum - f_slicenum) // 8 + 1, 8, i - f_slicenum + 1)  # TODO: Fix, adding extra line when X8
        plt.imshow(nii_slices[:, :, i], cmap='gray')
    plt.gcf().set_size_inches(30, 30)
    plt.gcf().tight_layout(pad=0)
    plt.show()


