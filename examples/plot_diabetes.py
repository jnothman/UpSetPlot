"""
==================================
Above-average features in Diabetes
==================================

Explore above-average attributes in the Diabetes dataset (Efron et al, 2004).

Here we take some features correlated with disease progression, and look at the
distribution of that disease progression value when each of these features is
above average.

The most correlated features are:

  - bmi body mass index
  - bp average blood pressure
  - s4 tch, total cholesterol / HDL
  - s5 ltg, possibly log of serum triglycerides level
  - s6 glu, blood sugar level

This kind of dataset analysis may not be a practical use of UpSet, but helps
to illustrate the :meth:`UpSet.add_catplot` feature.
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt
from upsetplot import UpSet

# Load the dataset into a DataFrame
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Get five features most correlated with median house value
correls = diabetes_df.corrwith(pd.Series(diabetes.target),
                             method='spearman').sort_values()
top_features = correls.index[-5:]

# Get a binary indicator of whether each top feature is above average
diabetes_above_avg = diabetes_df > diabetes_df.median(axis=0)
diabetes_above_avg = diabetes_above_avg[top_features]
diabetes_above_avg = diabetes_above_avg.rename(columns=lambda x: x + '>')

# Make this indicator mask an index of diabetes_df
diabetes_df = pd.concat([diabetes_df, diabetes_above_avg],
                      axis=1)
diabetes_df = diabetes_df.set_index(list(diabetes_above_avg.columns))

# Also give us access to the target (median house value)
diabetes_df = diabetes_df.assign(progression=diabetes.target)

##########################################################################

# UpSet plot it!
upset = UpSet(diabetes_df, subset_size='count', intersection_plot_elements=3)
upset.add_catplot(value='progression', kind='strip', color='blue')
print(diabetes_df)
upset.add_catplot(value='bmi', kind='strip', color='black')
upset.plot()
plt.title("UpSet with catplots, for orientation='horizontal'")
plt.show()

##########################################################################

# And again in vertical orientation

upset = UpSet(diabetes_df, subset_size='count', intersection_plot_elements=3,
              orientation='vertical')
upset.add_catplot(value='progression', kind='strip', color='blue')
upset.add_catplot(value='bmi', kind='strip', color='black')
upset.plot()
plt.title("UpSet with catplots, for orientation='vertical'")
plt.show()
