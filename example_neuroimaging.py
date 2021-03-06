from nilearn import datasets
dataset = datasets.fetch_haxby(n_subjects=1)

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_strategy='epi', smoothing_fwhm=6, memory='cache')

X_masked = masker.fit_transform(dataset.func[0])

from rena import ReNA
cluster = ReNA(scaling=True, n_clusters=2000, masker=masker)

X_reduced = cluster.fit_transform(X_masked[0: 10])
X_compressed = cluster.inverse_transform(X_reduced)


import numpy as np
# Shuffle the labels (for better visualization):
labels = cluster.labels_
permutation = np.random.permutation(labels.shape[0])
labels = permutation[labels]
labels_img_ = masker.inverse_transform(labels)

cut_coords = (-34, -16)
n_image = 0


import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_epi
plt.close('all')


clusters_fig = plot_stat_map(labels_img_, bg_img=dataset.anat[0],
                             title='clusters', display_mode='yz',
                             cut_coords=cut_coords, colorbar=False)

compress_fig = plot_epi(masker.inverse_transform(X_compressed[n_image]),
                        title='compressed', display_mode='yz',
                        cut_coords=cut_coords)

original_fig = plot_epi(masker.inverse_transform(X_masked[n_image]),
                        title='original', display_mode='yz',
                        cut_coords=cut_coords)


clusters_fig.savefig('figures/clusters.png')
compress_fig.savefig('figures/compress.png')
original_fig.savefig('figures/original.png')

plt.show()
