# pyright: ignore[reportMissingImports]
import numpy as np
import time
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from scipy.interpolate import splrep, splev, splint
import pickle

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid, z_grid, eps_grid,
    load_or_generate_B0arr
)


def main(input_bind, input_epsind):
    print(time.ctime())
    karr = k_grid()
    B0arr = load_or_generate_B0arr()
    zarr = z_grid()
    epsarr = eps_grid()

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    OUTBASE = PROJECT_ROOT / "data/outputs"

    cross_corrsave = OUTBASE / "cross_corr"
    ang_avg_Tb_dir = OUTBASE / f"ang_avg/Tb/B_{round(1e12*B0arr[input_bind])}pG"

    # Load existing cross_corr dict (produced by corrs_corr_and_source_fncs.py)
    with open(cross_corrsave / f'cross_corr_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'rb') as f:
        cross_corr = pickle.load(f)

    # Load Tb angle-averaged quantities for all k
    ang_avg_Tb_master = {}
    for k in [
        'TbTbbar', 'Tbxebar', 'Tbdeltambar', 'TbThetabar', 'deltamThetabar'
    ]:
        ang_avg_Tb_master[k] = np.zeros( ( len(karr), len(zarr) ) )

    for kind in range(len(karr)):
        with open(ang_avg_Tb_dir / f'ang_avg_k{kind}_Tb.pkl', 'rb') as f:
            ang_avg_Tb_dict = pickle.load(f)
        for k in [
            'TbTbbar', 'Tbxebar', 'Tbdeltambar', 'TbThetabar', 'deltamThetabar'
        ]:
            ang_avg_Tb_master[k][kind] = ang_avg_Tb_dict[k]

    # Initialize new keys
    for k in ['Tbrms', 'Tbxecross', 'Tbdeltacross', 'TbThetacross', 'deltaThetacross']:
        cross_corr[k] = np.zeros(len(zarr))

    for zind in range(len(zarr)):
        cross_corr['Tbrms'][zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) *
                    ang_avg_Tb_master['TbTbbar'][:,zind]
                )))

        cross_corr['Tbxecross'][zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) *
                    ang_avg_Tb_master['Tbxebar'][:,zind]
                )))

        cross_corr['Tbdeltacross'][zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) *
                    ang_avg_Tb_master['Tbdeltambar'][:,zind]
                )))

        cross_corr['TbThetacross'][zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) *
                    ang_avg_Tb_master['TbThetabar'][:,zind]
                )))

        cross_corr['deltaThetacross'][zind] = splint(
            karr[-1], karr[0], splrep(
                np.flip(karr), np.flip(
                    karr**(epsarr[input_epsind] - 1) *
                    ang_avg_Tb_master['deltamThetabar'][:,zind]
                )))

    # Re-save cross_corr with added Tb keys
    with open(cross_corrsave / f'cross_corr_B_{round(1e12*B0arr[input_bind])}pG_e{round(epsarr[input_epsind],3)}.pkl', 'wb') as f:
        pickle.dump(cross_corr, f)

    print(time.ctime())


if __name__ == '__main__':
    print(time.ctime())

    main(int(sys.argv[1]), int(sys.argv[2]))

    print(time.ctime())
