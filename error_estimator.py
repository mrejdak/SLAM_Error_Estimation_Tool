import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

def compute_signatures(pts, k=None):
    # pts: (N,2)
    N = pts.shape[0]
    D = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=2)  # NxN
    # zero diagonal -> remove self dist
    D_sorted = np.sort(D, axis=1)[:,1:]  # drop self (smallest)
    if k is None:
        return D_sorted  # full sorted distances as signature
    else:
        return D_sorted[:,:k]  # take k nearest distances

def hungarian_by_signature(slam_pts, real_pts, slam_colors, real_colors, k_sig=None, big=1e6):
    Ssig = compute_signatures(slam_pts, k=k_sig)
    Rsig = compute_signatures(real_pts, k=k_sig)
    N = slam_pts.shape[0]
    M = real_pts.shape[0]
    cost = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if slam_colors[i] != real_colors[j]:
                cost[i,j] = big
            else:
                # if signature length differs pad with large values
                a = Ssig[i]
                b = Rsig[j]
                L = min(a.shape[0], b.shape[0])
                cost[i,j] = np.linalg.norm(a[:L]-b[:L])
    row_ind, col_ind = linear_sum_assignment(cost)
    # return only pairs where cost < big
    mask = cost[row_ind, col_ind] < big/2
    matches = list(zip(row_ind[mask].tolist(), col_ind[mask].tolist()))
    return matches

def procrustes_transform(A, B):
    # A, B: (k,2) corresponding points
    assert A.shape == B.shape
    k = A.shape[0]
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Ac = (A - muA).T   # 2 x k
    Bc = (B - muB).T
    H = Ac.dot(Bc.T)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # ensure rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T.dot(U.T)
    t = muB - R.dot(muA)
    return R, t

def apply_transform(pts, R, t):
    return (R.dot(pts.T)).T + t

def match_and_estimate(slam_pts, real_pts, slam_colors, real_colors, k_sig=6, icp_iters=50, tol=1e-6):
    # initial matching
    matches = hungarian_by_signature(slam_pts, real_pts, slam_colors, real_colors, k_sig)
    if len(matches) < 3:
        print("not enough initial matches, need >=3 matched pairs")
        return
    A_idx = np.array([i for i,j in matches])
    B_idx = np.array([j for i,j in matches])
    A = slam_pts[A_idx]
    B = real_pts[B_idx]
    R, t = procrustes_transform(A, B)
    prev_err = np.inf

    # ICP refinement (color constrained)
    for it in range(icp_iters):
        S_tr = apply_transform(slam_pts, R, t)
        # build KD trees per color
        mapped_pairs = []
        for color in np.unique(real_colors):
            real_mask = np.array(real_colors) == color
            slam_mask = np.array(slam_colors) == color
            if real_mask.sum() == 0 or slam_mask.sum() == 0:
                continue
            tree = cKDTree(real_pts[real_mask])
            sims = S_tr[slam_mask]
            dists, idxs = tree.query(sims, k=1)
            # map back to original indices
            slam_idxs = np.where(slam_mask)[0]
            real_idxs = np.where(real_mask)[0][idxs]
            for si, ri, dd in zip(slam_idxs, real_idxs, dists):
                mapped_pairs.append((si, ri, dd))
        if len(mapped_pairs) < 3:
            break
        # take only unique pairs (one-to-one): resolve collisions by smallest distance
        # build dict real->(slam,dist) ensuring one-to-one
        chosen = {}
        for si, ri, dd in sorted(mapped_pairs, key=lambda x: x[2]):
            if ri not in chosen and si not in [v[0] for v in chosen.values()]:
                chosen[ri] = (si, dd)
        pairs = [(v[0], k) for k,v in chosen.items()]  # (slam_idx, real_idx)
        A = slam_pts[[p[0] for p in pairs]]
        B = real_pts[[p[1] for p in pairs]]
        R_new, t_new = procrustes_transform(A, B)
        S_tr_new = apply_transform(slam_pts, R_new, t_new)
        residuals = np.linalg.norm(S_tr_new[[p[0] for p in pairs]] - B, axis=1)
        err = np.sqrt((residuals**2).mean())
        if abs(prev_err - err) < tol:
            R, t = R_new, t_new
            break
        R, t = R_new, t_new
        prev_err = err

    # final correspondences: nearest same-color
    S_final = apply_transform(slam_pts, R, t)
    final_pairs = []
    for i,s_pt in enumerate(S_final):
        color = slam_colors[i]
        candidates = np.where(np.array(real_colors)==color)[0]
        if len(candidates)==0: continue
        d = np.linalg.norm(real_pts[candidates] - s_pt, axis=1)
        j = candidates[np.argmin(d)]
        final_pairs.append((i,j, d.min()))
    # RMS on matched pairs
    dists = np.array([p[2] for p in final_pairs])
    rms = np.sqrt((dists**2).mean()) if len(dists)>0 else np.nan
    return {
        'R': R, 't': t,
        'matches': final_pairs,
        'rms': rms,
        'residuals': dists
    }
