"""One-off: build brain.json for the report's 3D view (needs nibabel; the app itself does not).
ROI positions = centroids of the 360 Glasser parcels in MNI space (the atlas used to extract the ABIDE time series; column i = label i+1,
first 180 = right hemisphere, last 180 = left). Brain shell = fsaverage5 pial surface bundled with nilearn, every 2nd vertex.
'area' is a coarse lobe guessed from the centroid's coordinates, shown as 'approx.' in the report."""
import json, numpy as np, nibabel as nb
P = "/Users/patel/Desktop/COLLEGE/TY/SEM-5/ML/CP/GRAPH_ANAMOLY/"
ATLAS = P + "GSP-applied-on-ASD-classification/Glasser_masker.nii.gz"
MESH = P + ".venv/Lib/site-packages/nilearn/datasets/data/fsaverage5/"
im = nb.load(ATLAS); d = np.asarray(im.dataobj).round().astype(int)
labels = np.unique(d); labels = labels[labels > 0]; assert len(labels) == 360
cent = np.array([(im.affine @ np.r_[np.argwhere(d == l).mean(0), 1])[:3] for l in labels])
def area(x, y, z):
    if y < -62: return "occipital"
    if z < 8 and -72 <= y <= 18 and abs(x) > 26: return "temporal"
    if y < -20: return "parietal"
    if y >= -20 and z >= 0: return "frontal"
    return "deep / medial"
out = dict(roi=cent.round(1).tolist(), hemi=["right" if i < 180 else "left" for i in range(360)],
           area=[area(*c) for c in cent])
for side in ("left", "right"):
    v, _ = nb.load(MESH + f"pial_{side}.gii.gz").agg_data()
    out[side] = v[::2].round(1).tolist()
json.dump(out, open("brain.json", "w"), separators=(",", ":"))
print({k: len(v) for k, v in out.items()}, "x range", cent[:, 0].min(), cent[:, 0].max())
