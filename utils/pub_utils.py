import os, glob
import numpy as np
from main import get_GT, load_masks, calc_RMSE


def calcmeasures(S0,MD,p):
    S0_ = p[:, :, :, 0]
    MD_ = p[:, :, :, 1]

    S0_[S0_ < 0] = np.NAN
    MD_[S0_ < 0] = np.NAN
    S0_[MD_ < 0] = np.NAN
    MD_[MD_ < 0] = np.NAN




    masks = load_masks(filename="../../../masks.npy")
    masked_outliers = np.copy(S0_)
    masked_outliers[np.invert(masks)] = np.inf
    masked_outlinerN = np.count_nonzero(np.isnan(masked_outliers))

    S0_RMSE, mean_S0 = calc_RMSE(S0, S0_, masks)
    MD_RMSE, mean_MD = calc_RMSE(MD, MD_, masks)

    return S0_RMSE, mean_S0, MD_RMSE, mean_MD, masked_outlinerN


def gen_res_table(folder_path, f):
    os.chdir(folder_path)
    curr_l=-1
    for filename in sorted(glob.glob("p*.npy"),reverse=True):
        print(filename)
        l = int(filename[1: int(filename.find("_"))])
        filter_type = filename[int(filename.find("_"))+1:int(filename.find("."))]

        if l != curr_l:
            curr_l = l
            f.write("\\\ \hline")
            f.write("\n")
            f.write("\multirow{1}{*}{$\lambda="+str("{:.2f}".format(l/100))+"$}")
        else:
            f.write("\\\ \cline{2-7}")
            f.write("\n")

        f.write("& "+ filter_type)

        S0, MD = get_GT("../../simulation1", "Image_")
        p = np.load(filename)

        S0_RMSE, mean_S0, MD_RMSE, mean_MD, masked_outlinerN = calcmeasures(S0, MD, p)
        f.write("&" + str("{:.2f}".format(100 * masked_outlinerN / np.prod(S0.shape))))
        f.write("& " + str("{:.2f}".format(S0_RMSE))+" & " + str("{:.2f}".format(MD_RMSE)))
        f.write("& " + str("{:.2f}".format(100*S0_RMSE/mean_S0))+" & " + str("{:.2f}".format(100*MD_RMSE/mean_MD)))



if __name__ == '__main__':
    f = open('../params/res_params/newric_b100_800/res_table.text', 'w')
    gen_res_table("../params/res_params/newric_b100_800",f)
