# %%
import upsetplot as ups
import matplotlib.pyplot as plt

example = ups.generate_counts(seed=1, n_samples=1000, n_categories=4)

##########################################################################
# Axes can be styled similar to subplots, by name.
# ... with fill color

fig = plt.figure(dpi=400)

upset = ups.UpSet(example, sort_by="cardinality", min_subset_size=50)
# upset.style_axes(axes_names=["cat2"],
#                  facecolor=(0, 0, 1, 0.3))
upset.style_categories(category_names=["cat3"],
                       facecolor=(0, 1, 0, 0.3),
                       #  edgecolor="red",
                       #  linestyle="-",
                       linewidth=2)
upset.style_categories(category_names=["cat1"],
                       facecolor=(1, 0, 0, 0.3),
                       #  edgecolor="green",
                       #  linestyle="--",
                       linewidth=1)
upset.style_subsets(present=["cat1", "cat3"], facecolor="red")
plt.suptitle("Make a point...")
upset.plot(fig)
plt.savefig("styling.jpg", dpi=300)
