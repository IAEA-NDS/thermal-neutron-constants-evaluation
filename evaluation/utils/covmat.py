import numpy as np


def create_relative_covmat(data):
    """Calculate the full relative covariance matrix from Axton 1986 data.

    The relative covariance between measurements i and j is:
        cov_rel(i,j) = sum_k u_ik * u_jk
    where the sum runs over all correlation codes k shared by both
    measurements, and u_ik is the systematic uncertainty component
    (in percent) of measurement i for code k.

    The diagonal includes both the systematic components and the
    remaining uncorrelated uncertainty (added in quadrature).

    Parameters
    ----------
    data : dict
        Dictionary representation of the axton1986_data.json file.

    Returns
    -------
    covmat : np.ndarray
        n x n relative covariance matrix (in percent squared),
        where n is the number of measurements.
    measurement_nos : list of int
        The measurement serial numbers corresponding to each
        row/column of the matrix.
    """
    measurements = data["measurements"]
    n = len(measurements)
    measurement_nos = [m["no"] for m in measurements]

    syst_list = []
    total_unc = []
    for m in measurements:
        syst_list.append(m.get("systematic_uncertainties", {}))
        total_unc.append(m["uncertainty_percent"])

    covmat = np.zeros((n, n))

    # Off-diagonal and systematic part of diagonal:
    # collect all codes across all measurements for efficiency
    code_to_indices = {}
    for i, syst in enumerate(syst_list):
        for code in syst:
            if code not in code_to_indices:
                code_to_indices[code] = []
            code_to_indices[code].append(i)

    # For each code, add the outer product of its uncertainty vector
    for code, indices in code_to_indices.items():
        vals = np.array([syst_list[i][code] for i in indices])
        for a, ia in enumerate(indices):
            for b, ib in enumerate(indices):
                covmat[ia, ib] += vals[a] * vals[b]

    # Diagonal: total variance = total_unc^2
    # The systematic part is already on the diagonal from above.
    # The uncorrelated part fills the remainder.
    for i in range(n):
        covmat[i, i] = total_unc[i] ** 2

    return covmat, measurement_nos


def create_relative_covmat_from_ags(filepath):
    """Calculate the full relative covariance matrix from an AGS file.

    The AGS file format has one row per measurement with columns:
        col 0: measurement serial number
        col 1: weight (always 1.0)
        col 2: uncorrelated relative uncertainty (fraction)
        cols 3-66: systematic uncertainty components u_ik (fraction)

    The relative covariance is:
        cov_rel(i,j) = sum_k S[i,k] * S[j,k]  (off-diagonal)
        cov_rel(i,i) = uncorr_i^2 + sum_k S[i,k]^2  (diagonal)

    where S is the matrix of systematic components (cols 3-66).

    Parameters
    ----------
    filepath : str
        Path to an AGS file (.mac or .mic).

    Returns
    -------
    covmat : np.ndarray
        n x n relative covariance matrix (in fraction squared),
        where n is the number of measurements in the file.
    measurement_nos : list of int
        The measurement serial numbers corresponding to each
        row/column of the matrix.
    """
    measurement_nos = []
    uncorr = []
    syst_matrix = []

    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            measurement_nos.append(int(float(parts[0])))
            uncorr.append(float(parts[2]))
            syst_row = [float(x) for x in parts[3:]]
            syst_matrix.append(syst_row)

    uncorr = np.array(uncorr)
    S = np.array(syst_matrix)

    # cov = S @ S^T + diag(uncorr^2)
    covmat = S @ S.T + np.diag(uncorr ** 2)

    return covmat, measurement_nos


def load_thermalcst_data(*filepaths):
    """Load measurement data from thermalcst mac/mic files.

    Each file has one row per measurement with columns:
        col 0: measurement serial number
        col 1: measured value
        col 2: absolute uncertainty
        col 3: (always 0)

    Multiple files (e.g. mac + mic) are merged and sorted by
    measurement number.

    Parameters
    ----------
    *filepaths : str
        Paths to thermalcst files (.mac and/or .mic).

    Returns
    -------
    measurement_nos : list of int
        Measurement serial numbers (sorted).
    values : np.ndarray
        Measured values.
    uncertainties : np.ndarray
        Absolute uncertainties.
    """
    entries = {}
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                no = int(float(parts[0]))
                val = float(parts[1])
                unc = float(parts[2])
                entries[no] = (val, unc)

    measurement_nos = sorted(entries.keys())
    values = np.array([entries[n][0] for n in measurement_nos])
    uncertainties = np.array([entries[n][1] for n in measurement_nos])

    return measurement_nos, values, uncertainties


def create_relative_covmat_from_ags_combined(*filepaths):
    """Build a combined relative covariance matrix from multiple AGS files.

    Merges mac and mic AGS files into a single covariance matrix,
    sorted by measurement number. The files must not contain
    overlapping measurement IDs.

    Parameters
    ----------
    *filepaths : str
        Paths to AGS files (.mac and/or .mic).

    Returns
    -------
    covmat : np.ndarray
        n x n relative covariance matrix (in fraction squared).
    measurement_nos : list of int
        Measurement serial numbers (sorted).
    """
    all_entries = {}
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                no = int(float(parts[0]))
                if no in all_entries:
                    raise ValueError(
                        f"Duplicate measurement ID {no} across AGS files"
                    )
                uncorr = float(parts[2])
                syst = [float(x) for x in parts[3:]]
                all_entries[no] = (uncorr, syst)

    measurement_nos = sorted(all_entries.keys())
    n = len(measurement_nos)

    uncorr = np.array([all_entries[no][0] for no in measurement_nos])

    # Pad systematic vectors to the same length (mac and mic may differ)
    max_cols = max(len(all_entries[no][1]) for no in measurement_nos)
    S = np.zeros((n, max_cols))
    for i, no in enumerate(measurement_nos):
        row = all_entries[no][1]
        S[i, :len(row)] = row

    covmat = S @ S.T + np.diag(uncorr ** 2)

    return covmat, measurement_nos
