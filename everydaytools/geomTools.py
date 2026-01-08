import numpy as np

def order_curve_points(pts, approx_start_pt):
    s = np.linalg.norm(pts - approx_start_pt, axis=1)
    ordered = [np.argmin(s)]
    while len(ordered) < len(pts):
        s = np.linalg.norm(pts - pts[ordered[-1], :], axis=1)
        check_order = np.argsort(s)
        for i in check_order:
            if i not in ordered:
                ordered.append(i)
                break
    pts = pts[ordered, :]
    return pts