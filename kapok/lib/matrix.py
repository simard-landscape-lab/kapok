# -*- coding: utf-8 -*-
"""Matrix and linear algebra helper functions.

    Author: Michael Denbina

    Copyright 2016-2023 California Institute of Technology.  All rights reserved.
    United States Government Sponsorship acknowledged.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import collections

import numpy as np


def makehermitian(m):
    """Given a matrix with elements below the diagonal equal to zero, fill
        those elements assuming the matrix is Hermitian. Note: Changes array
        in place!  (But also returns it.)

        Arguments:
            m (array): Matrices to make Hermitian.  Should have dimensions:
                (az, rng, n, n).  n is the number of rows and columns in the
                matrix (which need to be equal).

        Returns:
            m (array): Hermitian symmetric form of input.

    """
    if m.ndim == 4:
        for row in range(1,m.shape[2]):
            for col in range(0,row):
                m[:,:,row,col] = np.conj(m[:,:,col,row])
    else:
        for row in range(1,m.shape[0]):
            for col in range(0,row):
                m[row,col] = np.conj(m[col,row])

    return m


def zerobelowdiag(m):
    """Given a Hermitian symmetric covariance matrix with all elements filled,
        set the redundant elements below the diagonal to zero, in order
        to save storage space.

        Note: Changes array in place!  (But also returns it.)

        Arguments:
            m (array): Matrices to cancel out the below diagonal elements.
                Should have dimensions: (az, rng, n, n).  n is the number of
                rows and columns in the matrix (which need to be equal).

        Returns:
            m (array): Compress form of input matrix.

    """
    if m.ndim == 4:
        for row in range(1,m.shape[2]):
            for col in range(0,row):
                m[:,:,row,col] = 0
    else:
        for row in range(1,m.shape[0]):
            for col in range(0,row):
                m[row,col] = 0

    return m


def convertT3toC3(m):
    """Change the polarization basis of a 3x3 Pauli coherency matrix
        to a 3x3 Lexicographic covariance matrix.

        Arguments:
            m (array): The matrices to change the basis of, with dimensions of
                (az, rng, 3, 3).  The third dimension contains the rows
                of the matrices, and the fourth dimension contains the columns.
                The matrices need to be 3x3.

        Returns:
            mc (array): The matrix with a Lexicographic polarization basis.

    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mc = np.zeros(x.shape, dtype=x.dtype)

    srt = np.sqrt(2)

    mc[0,0] = x[0,0] + x[1,0] + x[0,1] + x[1,1]
    mc[0,1] = srt*(x[0,2] + x[1,2])
    mc[0,2] = x[0,0] + x[1,0] - x[0,1] - x[1,1]
    mc[1,0] = srt*(x[2,0] + x[2,1])
    mc[1,1] = 2*x[2,2]
    mc[1,2] = srt*(x[2,0]-x[2,1])
    mc[2,0] = x[0,0] - x[1,0] + x[0,1] - x[1,1]
    mc[2,1] = srt*(x[0,2] - x[1,2])
    mc[2,2] = x[0,0] - x[1,0] - x[0,1] + x[1,1]

    mc /= 2
    mc = np.rollaxis(np.rollaxis(mc,-1),-1)
    return mc


def convertC3toT3(m):
    """Change the polarization basis of a 3x3 Lexicographic covariance matrix
        to a 3x3 Pauli coherency matrix.

        Arguments:
            m (array): The matrices to change the basis of, with dimensions of
                (az, rng, 3, 3).  The third dimension contains the rows
                of the matrices, and the fourth dimension contains the columns.
                The matrices need to be 3x3.

        Returns:
            mc (array): The matrix with a Pauli polarization basis.

    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mc = np.zeros(x.shape, dtype=x.dtype)

    srt = np.sqrt(2)

    mc[0,0] = x[0,0] + x[2,0] + x[0,2] + x[2,2]
    mc[0,1] = x[0,0] + x[2,0] - x[0,2] - x[2,2]
    mc[0,2] = srt*(x[0,1] + x[2,1])
    mc[1,0] = x[0,0] - x[2,0] + x[0,2] - x[2,2]
    mc[1,1] = x[0,0] - x[2,0] - x[0,2] + x[2,2]
    mc[1,2] = srt*(x[0,1] - x[2,1])
    mc[2,0] = srt*(x[1,0] + x[1,2])
    mc[2,1] = srt*(x[1,0] - x[1,2])
    mc[2,2] = 2*x[1,1]

    mc /= 2
    mc = np.rollaxis(np.rollaxis(mc,-1),-1)
    return mc


def rotateT3(m, psi):
    """Given a 3x3 Pauli basis coherency matrix and a rotation angle, apply
        a rotation in the polarization plane.

        Arguments:
            m: An array containing the input coherency matrices, with
                dimensions of (az, rng, 3, 3).
            psi: The desired rotation angle, in radians.

        Returns:
            mr: The coherency matrix rotated by psi.

    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mr = np.zeros(x.shape, dtype=x.dtype)

    c = np.cos(2*psi)
    s = np.sin(2*psi)

    mr[0,0] = x[0,0]
    mr[0,1] = x[0,1]*c + x[0,2]*s
    mr[0,2] = x[0,1]*(-s) + x[0,2]*c
    mr[1,0] = x[1,0]*c + x[2,0]*s
    mr[1,1] = c*(x[1,1]*c + x[2,1]*s) + s*(x[1,2]*c + x[2,2]*s)
    mr[1,2] = (-s)*(x[1,1]*c + x[2,1]*s) + c*(x[1,2]*c + x[2,2]*s)
    mr[2,0] = x[1,0]*(-s) + x[2,0]*c
    mr[2,1] = c*(x[1,1]*(-s) + x[2,1]*c) + s*(x[1,2]*(-s) + x[2,2]*c)
    mr[2,2] = (-s)*(x[1,1]*(-s) + x[2,1]*c) + c*(x[1,2]*(-s) + x[2,2]*c)

    mr = np.rollaxis(np.rollaxis(mr,-1),-1)
    return mr


def linesegmentdist(p, w, v, full_line=False):
    """Function to calculate the shortest line length between a complex
    coherence and a line segment in the complex plane.

    Arguments:
        p: The coherence point (e.g., the modelled volume coherence, or
            the origin).
        w: One end of the line segment (e.g., the observed high
            coherence).
        v: The other point on the line segment (e.g., the observed low
            coherence).
        full_line (bool): Set to True if you do not want to check just the
            line segment, but rather the entire line through w and v, for
            the solution.  Default: False (constrain the projected point
            to lie between w and v).

    Returns:
        line:  The shortest line length.  We take the distance of this to
            calculate the cost for the coherences.

    """
    l2 = np.square(np.abs(w - v)) # Squared length of line segment.

    if np.all(l2 == 0): # Check if line segment has zero length.
        return np.abs(p - v)
    else:
        # Consider line segment parameterized as v + t(w - v).  Project
        # p onto this line, but clip t bounded to [0,1].
        t = (np.real(p-v)*np.real(w-v) + np.imag(p-v)*np.imag(w-v)) / l2
        if not full_line:
            t = np.clip(t, 0, 1)
        proj = v + t*(w-v)
        return np.abs(p - proj)


def spcov(m, pol=0):
    """Function to return a single-polarization covariance matrix from the
    full multi-polarization covariance matrix.

    Arguments:
        m (array): The fully-polarimetric covariance matrix, with dimensions
            (azimuth, range, n*3, n*3) (where n is the number of flight tracks).
        pol: An integer polarization integer id, or a string, or a weighting
            vector with dimensions (3).

    Returns:
        m (array): The single-polarization covariance matrix, with dimensions
            (azimuth, range, n, n) (where n is the number of flight tracks).

    """
    # TO-DO: Implement polarimetric weight vectors.
    pass
    # Polarization input can be a string, e.g., "HH", "HV", or "VV"
    if isinstance(pol,str):
        pol = pol.lower()

    if isinstance(pol,str) and (pol == 'hh'):
        pol = 0
    elif isinstance(pol,str) and (pol == 'hv'):
        pol = 1
    elif isinstance(pol,str) and (pol == 'vv'):
        pol = 2
    elif isinstance(pol,str) and (pol == 'hh+vv'):
        pol = [np.sqrt(0.5), 0, np.sqrt(0.5)]
    elif isinstance(pol,str) and (pol == 'hh-vv'):
        pol = [np.sqrt(0.5), 0, -np.sqrt(0.5)]

    # Polarization identifier is a scalar index:
    if not isinstance(pol, (collections.abc.Sequence, np.ndarray)):
        spcov = m[:,:,pol::3,pol::3]
        return spcov
    else: # Polarization identifier is a weighting vector.
        # TO-DO: NYI
        i,j = mb_cov_index(bl, pol=0, n_pol=self.num_pol)
        return


def normalizematrix(m):
    """ Normalize an input covariance matrix for the backscattered power in
        different tracks and polarizations using the inverse square root of
        the diagonalization.

        Arguments:
            m (array): The fully-polarimetric covariance matrix, with dimensions
                (azimuth, range, n, n) (where n is the number of rows and
                columns of the matrix).

        Returns:
            m_norm (array): The normalized covariance matrix, with same
                dimensions as the input.

    """
    m_diag = 1 / np.sqrt(np.diagonal(m, axis1=2, axis2=3).copy())
    m_diag = m_diag[..., np.newaxis] * np.eye(m.shape[2])[np.newaxis, np.newaxis, ...]
    m_norm = m_diag @ m @ m_diag
    return m_norm


def rearrange_for_skp(m):
    """ Vectorize the input covariance matrix for use in the SKP decomposition,
    as in the Appendix of this paper:

    S. Tebaldini, "Algebraic Synthesis of Forest Scenarios From
    Multibaseline PolInSAR Data," in IEEE Transactions on Geoscience and
    Remote Sensing, vol. 47, no. 12, pp. 4132-4142, Dec. 2009,
    doi: 10.1109/TGRS.2009.2023785.

    Arguments:
        m (array): The fully-polarimetric covariance matrix, with dimensions
            (azimuth, range, n*3, n*3) (where n is the number of flight
            tracks).

    Returns:
        m_rearr (array): Rearranged form of m, with dimensions
                (azimuth, range, n*n, 9) (where n is the number of flight
                tracks).

    """
    n_pol = 3
    n_az = m.shape[0]
    n_rg = m.shape[1]
    n_tr = m.shape[2] // n_pol

    m_rearr = np.zeros((m.shape[0], m.shape[1], n_tr**2, n_pol**2), dtype='complex64')

    for tr1 in np.arange(n_tr):
        for tr2 in np.arange(n_tr):
            cov_row = tr1*n_pol
            cov_col = tr2*n_pol
            sub_m = m[:,:, cov_row:cov_row+n_pol, cov_col:cov_col+n_pol]
            sub_m = np.reshape(sub_m, (n_az, n_rg, n_pol**2), order='C')
            rearr_tr_index = tr1*n_tr + tr2
            m_rearr[:, :, rearr_tr_index, :] = sub_m

    return m_rearr


def skp_decomposition(m, k=2):
    """ Perform the SKP decomposition on a multi-baseline multi-polarimetric
    covariance matrix as described in this paper:

    S. Tebaldini, "Algebraic Synthesis of Forest Scenarios From
    Multibaseline PolInSAR Data," in IEEE Transactions on Geoscience and
    Remote Sensing, vol. 47, no. 12, pp. 4132-4142, Dec. 2009,
    doi: 10.1109/TGRS.2009.2023785.

    Arguments:
        m (array): The fully-polarimetric covariance matrix, with dimensions
            (azimuth, range, n*3, n*3) (where n is the number of flight
            tracks).
        k (int): Number of different scattering mechanisms to return from the
            SKP decomposition.

    Returns:
        ctilde (array): The candidate polarimetric covariance matrices for the
            data given by the SKP decomposition, with dimensions
            (azimuth, range, k, 3, 3)
        rtilde (array): The candidate interferometric structure matrices for the
            data given by the SKP decomposition, with dimensions
            (azimuth, range, k, n, n) (where n is the number of flight tracks).

    """
    n_pol = 3
    n_az = m.shape[0]
    n_rg = m.shape[1]
    n_tr = m.shape[2] // n_pol

    # Normalize covariance matrix by signal power in different tracks/polarizations:
    m_norm = normalizematrix(m)

    # Rearrange matrices to have dimensions (n_pol**2, n_tr**2) rather than
    # (n_pol*n_tr, n_pol*n_tr):
    m_rearr = rearrange_for_skp(m_norm)

    u, s, vh = np.linalg.svd(m_rearr, full_matrices=False)

    ctilde = np.zeros((k, m.shape[0], m.shape[1], n_pol, n_pol), dtype='complex64')
    rtilde = np.zeros((k, m.shape[0], m.shape[1], n_tr, n_tr), dtype='complex64')

    # Calculate candidate C and R matrices from SVD decomposition vectors u and
    # vh, scaled by singular values:
    # (Idea is that sum of Kronecker products of C and R matrices will produce
    # approximated multi-baseline, multi-polarimetric covariance matrix).
    for num in np.arange(k):
        ctilde[num] = (s[:, :, num, np.newaxis] * (vh[:, :, num])).reshape(n_az, n_rg, n_pol, n_pol, order = 'C')
        rtilde[num] = u[..., num].reshape(n_az, n_rg, n_tr, n_tr, order = 'C')

    # Scale C and R matrices so that R has upper-left element equal to 1.
    # This is done because we want all backscatter information to be
    # contained in the C matrix, the R matrix is for interferometric structure
    # only, while the C matrix is for the polarimetric covariance.
    scale_factor = rtilde[..., 0, 0]
    ctilde *= scale_factor[..., np.newaxis, np.newaxis]
    rtilde /= scale_factor[..., np.newaxis, np.newaxis]

    return ctilde, rtilde


def check_matrix_eigenvalues(m, allow_sign_flip=False):
    """ Helper function to check input matrices for semipositive definiteness.

    Arguments:
        m (array): The input matrices to check.  This function assumes the
            input matrix is Hermitian, so if the input matrix is not Hermitian,
            this function will fail.

    Returns:
        valid (array): Boolean array containing True for each index where the
            input matrix is semipositive definite (all eigenvalues are real and
            greater than or equal to zero).
    """
    eig = np.linalg.eigvalsh(m)

    ind_valid = np.all(eig >= 0, axis=-1)

    if allow_sign_flip:
        # Allow all matrices to be less than or equal to zero, i.e., in the
        # case where we are not sure how the matrix is scaled yet, and it could
        # be multiplied by a negative number.
        ind_valid = ind_valid | np.all(eig <= 0, axis=-1)

    return ind_valid


def mix_skp_matrices(ctilde, rtilde, a=None, b=None, check_matrices=False):
    """ Helper function to implement equations (24) and (25) from this paper:

    S. Tebaldini, "Algebraic Synthesis of Forest Scenarios From
    Multibaseline PolInSAR Data," in IEEE Transactions on Geoscience and
    Remote Sensing, vol. 47, no. 12, pp. 4132-4142, Dec. 2009,
    doi: 10.1109/TGRS.2009.2023785.

    Arguments:
        ctilde (array): The candidate polarimetric covariance matrices for the
            data given by the SKP decomposition, with dimensions
            (azimuth, range, k, 3, 3)
        rtilde (array): The candidate interferometric structure matrices for the
            data given by the SKP decomposition, with dimensions
            (azimuth, range, k, n, n) (where n is the number of flight tracks).
        a (array): a values with dimensions (azimuth, range)
        b (array): b values with dimensions (azimuth, range)
        check_matrices (bool): Set to True to check matrices to ensure they are
            semi-positive definite before returning them.  Any matrices which
            are not semi-positive definite will be set to entirely NaN values.

    """
    cmat = np.ones_like(ctilde) * np.nan
    rmat = np.ones_like(rtilde) * np.nan

    if (a is None):
        rmat[1] = b[..., np.newaxis, np.newaxis] * rtilde[0] + (1 - b[..., np.newaxis, np.newaxis]) * rtilde[1]
        cmat[0] = (1 - b[..., np.newaxis, np.newaxis]) * ctilde[0] - b[..., np.newaxis, np.newaxis] * ctilde[1]

        if check_matrices:
            ind_valid = check_matrix_eigenvalues(rmat[1])
            if np.any(~ind_valid):
                rmat[1][~ind_valid] = np.nan

            ind_valid = check_matrix_eigenvalues(cmat[0], allow_sign_flip=True)
            if np.any(~ind_valid):
                cmat[0][~ind_valid] = np.nan

    elif (b is None):
        rmat[0] = a[..., np.newaxis, np.newaxis] * rtilde[0] + (1 - a[..., np.newaxis, np.newaxis]) * rtilde[1]
        cmat[1] = -(1 - a[..., np.newaxis, np.newaxis]) * ctilde[0] + a[..., np.newaxis, np.newaxis] * ctilde[1]

        if check_matrices:
            ind_valid = check_matrix_eigenvalues(rmat[0])
            if np.any(~ind_valid):
                rmat[0][~ind_valid] = np.nan

            ind_valid = check_matrix_eigenvalues(cmat[1], allow_sign_flip=True)
            if np.any(~ind_valid):
                cmat[1][~ind_valid] = np.nan
    else:
        # Both a and b parameters specified.
        rmat[0] = a[..., np.newaxis, np.newaxis] * rtilde[0] + (1 - a[..., np.newaxis, np.newaxis]) * rtilde[1]
        rmat[1] = b[..., np.newaxis, np.newaxis] * rtilde[0] + (1 - b[..., np.newaxis, np.newaxis]) * rtilde[1]

        ab_scale = 1 / (a - b)
        cmat[0] = ab_scale[..., np.newaxis, np.newaxis] * ((1 - b[..., np.newaxis, np.newaxis]) * ctilde[0] - b[..., np.newaxis, np.newaxis] * ctilde[1])
        cmat[1] = ab_scale[..., np.newaxis, np.newaxis] * (-(1 - a[..., np.newaxis, np.newaxis]) * ctilde[0] + a[..., np.newaxis, np.newaxis] * ctilde[1])

        if check_matrices:
            ind_valid = check_matrix_eigenvalues(rmat[0])
            if np.any(~ind_valid):
                rmat[0][~ind_valid] = np.nan

            ind_valid = check_matrix_eigenvalues(rmat[1])
            if np.any(~ind_valid):
                rmat[1][~ind_valid] = np.nan

            ind_valid = check_matrix_eigenvalues(cmat[0])
            if np.any(~ind_valid):
                cmat[0][~ind_valid] = np.nan

            ind_valid = check_matrix_eigenvalues(cmat[1])
            if np.any(~ind_valid):
                cmat[1][~ind_valid] = np.nan

    return cmat, rmat


def reconstruct_skp_matrices(cmat, rmat):
    """ Reconstruct SKP decomposed matrices into original multi-polarization
    multi-baseline covariance matrix, as described in this paper:

    S. Tebaldini, "Algebraic Synthesis of Forest Scenarios From
    Multibaseline PolInSAR Data," in IEEE Transactions on Geoscience and
    Remote Sensing, vol. 47, no. 12, pp. 4132-4142, Dec. 2009,
    doi: 10.1109/TGRS.2009.2023785.

    Arguments:
        cmat (array): The polarimetric covariance matrices given by the SKP
            decomposition, with dimensions (azimuth, range, k, 3, 3).
        rmat (array): The interferometric structure matrices given by the SKP
            decomposition, with dimensions (azimuth, range, k, n, n)
            (where n is the number of flight tracks).

    Returns:
        m (array): The fully-polarimetric multi-baseline covariance matrix,
            with dimensions (azimuth, range, n*3, n*3) (where n is the number
            of flight tracks), reconstructed from the SKP decomposed matrices
            cmat and rmat.

    """



def maximum_structure_separation_skp_decomposition(m):
    """ Perform the SKP decomposition on the input covariance matrix,
        returning the C (polarimetric covariance) and R (structure) matrices
        for two scattering mechanisms which are both physically valid and
        have the maximum separation, as described in this paper:

        S. Tebaldini, "Algebraic Synthesis of Forest Scenarios From
        Multibaseline PolInSAR Data," in IEEE Transactions on Geoscience and
        Remote Sensing, vol. 47, no. 12, pp. 4132-4142, Dec. 2009,
        doi: 10.1109/TGRS.2009.2023785.

        Arguments:
            m (array): The fully-polarimetric covariance matrix, with dimensions
                (azimuth, range, n*3, n*3) (where n is the number of flight
                tracks).

        Returns:
            TO-DO

    """
    n_az = m.shape[0]
    n_rg = m.shape[1]

    ctilde, rtilde = skp_decomposition(m, k=2)

    # Bounds and sampling of the search space for the a and b parameters
    # which are used to form the C and R matrices from the SKP decomposition
    # results.
    a_search_min = np.zeros((n_az, n_rg), dtype='float32')
    b_search_min = np.zeros((n_az, n_rg), dtype='float32')
    a_search_max = np.ones((n_az, n_rg), dtype='float32')
    b_search_max = np.ones((n_az, n_rg), dtype='float32')
    search_step = 0.1
    current_a = a_search_min.copy()
    current_b = b_search_min.copy()
    iter_num = 0

    # Arrays to store the best values of a, b, and the objective function so far.
    best_a = np.ones((n_az, n_rg), dtype='float32') * np.nan
    best_b = np.ones((n_az, n_rg), dtype='float32') * np.nan
    best_objective = np.ones((n_az, n_rg), dtype='float32') * np.inf

    while iter_num < 10: # max 10 iterations
        while np.all((current_a <= a_search_max) & (current_b <= b_search_max)):
            pass

    for a in a_search:
        for b in b_search:
            # Check to make sure alpha matrix formed by a and b coefficients is
            # non-singular.  For a 2x2 matrix, we can check to make sure
            # the determinant of alpha is non-zero.
            det = a*(1-b) - (1-a)*b

            ctest, rtest = mix_skp_matrices(ctilde, rtilde, a, b)

    pass
