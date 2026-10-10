"""The frozen map function, shared by the build (bulk, faiss) and the server (single queries, numpy).

A paper lands at the similarity-weighted mean of the map positions of its K most similar
reference papers (cosine in the original 768-D DenseOn space), weights softmax(sim / TAU).
References, their coordinates, K and TAU are fixed per build, so a paper's position never
depends on which other papers are being shown.
"""
import numpy as np

K = 10
TAU = 0.01


def weights(sims):
    w = np.exp((sims - sims[:, :1]) / TAU)
    return w / w.sum(1, keepdims=True)


def place_numpy(Xref, Yref, q):
    """q: float32 [n, d] normalized. Exact; fine for a handful of queries."""
    sims = q @ Xref.T
    nn = np.argpartition(-sims, K, axis=1)[:, :K]
    s = np.take_along_axis(sims, nn, 1)
    o = np.argsort(-s, 1)
    nn, s = np.take_along_axis(nn, o, 1), np.take_along_axis(s, o, 1)
    return (weights(s)[:, :, None] * Yref[nn]).sum(1).astype(np.float32)


def place_faiss(Xref, Yref, X, exclude=None, batch=200_000, log=print):
    """Exact inner-product search over all references; exclude[i] = reference row to skip (or -1)."""
    import faiss
    index = faiss.IndexFlatIP(Xref.shape[1])
    index.add(np.ascontiguousarray(Xref, dtype=np.float32))
    out = np.empty((len(X), 2), np.float32)
    for a in range(0, len(X), batch):
        q = np.ascontiguousarray(X[a:a + batch], dtype=np.float32)
        sims, nn = index.search(q, K + 1)
        ex = exclude[a:a + batch] if exclude is not None else np.full(len(q), -1)
        keep = nn != ex[:, None]
        keep &= np.cumsum(keep, 1) <= K
        sims = sims[keep].reshape(len(q), K)
        nn = nn[keep].reshape(len(q), K)
        out[a:a + batch] = (weights(sims)[:, :, None] * Yref[nn]).sum(1)
        log(f'placed {min(a + batch, len(X)):,}/{len(X):,}')
    return out


# ---------------------------------------------------------------------------------------------
# Rule "mode" (atlas builds from 2026-10-09 on: their reference.npz has rule == "mode").
#
# The mean of the k nearest references (above) fails in two ways once tens of millions of
# papers are placed against 291,556 references:
#  - a paper whose nearest references sit in two places lands between them, in empty space
#    (17.7% of held-out papers landed farther than 3 reference spacings from any reference);
#  - with tau = 0.01 the nearest reference takes almost all the weight, so ~10% of papers sit
#    exactly on a reference: bright pinpoints and streaks when zoomed in.
# "mode" keeps the k = 30 nearest references, weights softmax(sim / 0.05), and puts the paper at
# the weighted mean of the group of references around the densest of them (a few mean-shift
# steps with a Gaussian of width h). Then a small offset that is fixed per W-number (Gaussian,
# sd = JITTER x the spacing of the nearest of those references around the result) spreads
# papers over the reference's neighbourhood instead of piling them on one point: positions
# finer than the reference spacing carry no information either way. Measured on 150,000
# held-out papers (share of the 10 nearest neighbours in meaning that are also among the 10
# nearest on the map): 0.123 -> 0.206; stranded in empty space 17.7% -> 0.8%; exactly on a
# reference 10.4% -> 0.8%. A question (no W-number) gets no offset.
MODE_K, MODE_TAU, MODE_ITERS = 30, 0.05, 3
SALT_R, SALT_A = np.uint64(0xA24BAED4963EE407), np.uint64(0x9FB21C651E98DF25)


def splitmix_u(ids):
    """Deterministic uniform [0,1) per uint64 (same finalizer as sample.splitmix_u)."""
    with np.errstate(over='ignore'):
        z = np.asarray(ids).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def offsets(works):
    """Unit Gaussian 2-D offset fixed per W-number (Box-Muller on two salted hashes)."""
    w = np.asarray(works).astype(np.uint64)
    u1 = np.maximum(splitmix_u(w ^ SALT_R), 1e-300)
    u2 = splitmix_u(w ^ SALT_A)
    r = np.sqrt(-2.0 * np.log(u1))
    return np.stack([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)], 1)


def mode_place(sims, nn, Yref, spacing, tau=MODE_TAU, h=7.6e-4, jitter=0.25, works=None, iters=MODE_ITERS):
    """sims, nn: [n, k] nearest references, best first. Yref [R, 2]; spacing [R] (distance from
    each reference to its nearest other reference on the map). Returns float32 [n, 2]."""
    sims = np.asarray(sims, np.float64)
    w = np.exp((sims - sims[:, :1]) / tau)
    w /= w.sum(1, keepdims=True)
    Y = np.asarray(Yref, np.float64)[nn]                                    # n, k, 2
    d2 = ((Y[:, :, None, :] - Y[:, None, :, :]) ** 2).sum(-1)              # n, k, k
    g = np.exp(-d2 / (2 * h * h))
    j = (w[:, :, None] * g).sum(1).argmax(1)                               # densest reference
    rows = np.arange(len(Y))
    c = Y[rows, j]
    for _ in range(iters):
        kk = w * np.exp(-((Y - c[:, None]) ** 2).sum(-1) / (2 * h * h))
        c = (kk[:, :, None] * Y).sum(1) / kk.sum(1, keepdims=True)
    if works is not None and jitter > 0:
        near = ((Y - c[:, None]) ** 2).sum(-1).argmin(1)                    # nearest of the k to c
        sd = jitter * np.asarray(spacing, np.float64)[nn[rows, near]]
        c = c + offsets(works) * sd[:, None]
    return np.clip(c, 0.0, 1.0).astype(np.float32)


class Rule:
    """The placement rule of an atlas build, from its reference.npz (old builds: mean)."""

    def __init__(self, ref):
        self.k, self.tau = int(ref['k']), float(ref['tau'])
        self.kind = str(ref['rule']) if 'rule' in ref else 'mean'
        self.Y = np.asarray(ref['coords'], np.float32)
        if self.kind == 'mode':
            self.h, self.jitter = float(ref['h']), float(ref['jitter'])
            self.spacing = np.asarray(ref['spacing'], np.float32)

    def __call__(self, sims, nn, works=None):
        if self.kind == 'mode':
            return mode_place(sims, nn, self.Y, self.spacing, self.tau, self.h, self.jitter, works)
        w = np.exp((np.asarray(sims, np.float64) - np.asarray(sims, np.float64)[:, :1]) / self.tau)
        w /= w.sum(1, keepdims=True)
        return (w[:, :, None] * self.Y[nn]).sum(1).astype(np.float32)
