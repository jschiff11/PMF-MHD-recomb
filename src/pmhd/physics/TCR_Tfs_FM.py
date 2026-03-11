import numpy as np
import time
import sys
from pathlib import Path

from scipy.integrate import solve_ivp
from scipy.interpolate import splrep, splev

from pmhd import cons, pars
from pmhd.data.grids import (
    k_grid,
    theta_grid,
    load_or_generate_z_arrays,
    load_or_generate_B0arr,
)


def main(input_bind, input_kind):
    print(time.ctime())
    karr = k_grid()
    thetaarr = theta_grid()

    zcrossarr, zfsarr = load_or_generate_z_arrays()
    B0arr = load_or_generate_B0arr()

    zstart = 2300
    xepeebs = pars.odeint(pars.RHSsob, pars.sahataylor(zstart), np.arange(zstart, 200, -1))
    xesaha = pars.sahataylor(np.arange(3e4, zstart, -1))
    xesaha2 = pars.sahataylor(np.arange(3e5, 2660, -1))
    xesaha3 = pars.sahapol(np.arange(2660, 200, -1))

    ## Compute ionization fraction as given by TLA and by saha. xe_full(z) assumes saha for z > 2300 then TLA after, xesaha_full assumes saha for all redshifts

    xe_hold = splrep(
        np.flip(np.concatenate([np.linspace(1e11, 3.1e4, 10000), np.arange(3e4, 200, -1)])),
        np.flip(np.concatenate([np.ones(10000), xesaha, xepeebs[:, 0]]))
    )

    xesaha_hold = splrep(
        np.flip(np.concatenate([np.linspace(1e11, 3.1e5, 10000), np.arange(3e5, 200, -1)])),
        np.flip(np.concatenate([np.ones(10000), xesaha2, xesaha3]))
    )

    def xe_full(z):
        return splev(z, xe_hold)
    def xesaha_full(z):
        return splev(z, xesaha_hold)

    def TCRmaginteg(karr, thetaarr, B0arr, kind, bind, thetaind, zstart, zend):
        sol = solve_ivp(pars.TCmag, [zstart, zend], [1, 0, 0, 0], args=(
                    karr[kind], thetaarr[thetaind], B0arr[bind], xe_full), method='LSODA',
                    dense_output=True, atol=1e-8, rtol=1e-6)
        return sol.sol(np.logspace(np.log10(zstart), np.log10(zend), num=10**4))

    print(time.ctime())

    resultsmag = np.zeros((len(thetaarr), 4, 10**4))
    for thetaind in range(len(thetaarr)):
        resultsmag[thetaind, :, :] = TCRmaginteg(karr, thetaarr, B0arr,
            input_kind, input_bind, thetaind, zcrossarr[input_kind], zfsarr[input_kind])

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    OUTBASE = PROJECT_ROOT / "data/outputs/Tfs/FM"

    Bdir = OUTBASE / f"B_{round(1e12*B0arr[input_bind])}pG"
    Bdir.mkdir(parents=True, exist_ok=True)

    np.save(Bdir / f"TCRmag_k{input_kind}.npy", resultsmag)

    print(time.ctime())


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
