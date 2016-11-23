import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, show
from nilearn.input_data import NiftiMasker
from rena import ReNA

# config
n_jobs = int(os.environ.get("N_JOBS", 1))
root = os.environ.get("ROOT", "/")
mem = os.environ.get("CACHE_DIR", "nilearn_cache")
mask_img = os.path.join(root, "storage/data/HCP_extra/mask_img.nii.gz")
n_clusters = int(os.environ.get("N_CLUSTERS", 100))
smoothing_fwhm = float(os.environ.get("FHWM", 6.))

# get data to be parcellated
zmaps = sorted(
    glob.glob(os.path.join(root, "storage/data/HCP/S500-*/*/MNINonLinear",
                           "Results/tfMRI_LANGUAGE/",
                           "tfMRI_LANGUAGE_hp200_s4_level2vol.feat/",
                           "cope4.feat/stats/zstat1.nii.gz")))

masker = NiftiMasker(mask_strategy='epi', smoothing_fwhm=smoothing_fwhm,
                     memory=mem, mask_img=mask_img)

X_masked = masker.fit_transform(zmaps)

model = ReNA(scaling=True, n_clusters=n_clusters, masker=masker,
             memory=mem)
X_reduced = model.fit_transform(X_masked)
X_compressed = model.inverse_transform(X_reduced)
masker.inverse_transform(X_compressed).to_filename("compressed.nii.gz")

# shuffle the labels (for better visualization):
labels = model.labels_ + 1
labels_img_ = masker.inverse_transform(labels)
labels_img_.to_filename("hcp_zmap_parcels.nii.gz")

# plot stuff
cut_coords = (-52, -2)
display_mode = "yx"
n_image = 0
plt.close('all')
clusters_slicer = plot_stat_map(labels_img_, title='clusters',
                                display_mode=display_mode,
                                cut_coords=cut_coords, colorbar=False)

compress_slicer = plot_stat_map(
    masker.inverse_transform(X_compressed[n_image]),
    title='compressed', display_mode=display_mode,
    cut_coords=cut_coords)

original_slicer = plot_stat_map(masker.inverse_transform(X_masked[n_image]),
                                title='original', display_mode=display_mode,
                                cut_coords=cut_coords)

clusters_slicer.savefig('figures/hcp_zmap_clusters.png')
compress_slicer.savefig('figures/hcp_zmap_compress.png')
original_slicer.savefig('figures/hcp_zmap_original.png')
show()
