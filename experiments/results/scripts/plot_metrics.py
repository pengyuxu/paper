import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load file of compiled metrics into a dataframe

df_metrics_all_authors = pd.read_csv("metrics_all_authors.csv")

#################################################
#           Make plot for accuracy              #
#################################################
ax = sns.barplot(
    df_metrics_all_authors,
    x="Method",
    y="Accuracy (%)",
    hue="Method Type",
    palette=["orangered", "mediumvioletred", "lightseagreen"],
)
ax.tick_params(axis="x", labelrotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
plt.savefig("accuracy_all.png", format="png", bbox_inches="tight", dpi=400)
plt.close()

#################################################
#        Make plot for balanced accuracy        #
#################################################
ax = sns.barplot(
    df_metrics_all_authors,
    x="Method",
    y="Balanced Accuracy (%)",
    hue="Method Type",
    palette=["orangered", "mediumvioletred", "lightseagreen"],
)
ax.tick_params(axis="x", labelrotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
plt.savefig("balanced_accuracy_all.png", format="png", bbox_inches="tight", dpi=400)
plt.close()

#################################################
#           Make plot for F1 score              #
#################################################
ax = sns.barplot(
    df_metrics_all_authors,
    x="Method",
    y="F1 Score (%)",
    hue="Method Type",
    palette=["orangered", "mediumvioletred", "lightseagreen"],
)
ax.tick_params(axis="x", labelrotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
plt.savefig("f1_all.png", format="png", bbox_inches="tight", dpi=400)
plt.close()

#################################################
#           Make plot for precision             #
#################################################
ax = sns.barplot(
    df_metrics_all_authors,
    x="Method",
    y="Precision (%)",
    hue="Method Type",
    palette=["orangered", "mediumvioletred", "lightseagreen"],
)
ax.tick_params(axis="x", labelrotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
plt.savefig("precision_all.png", format="png", bbox_inches="tight", dpi=400)
plt.close()

#################################################
#             Make plot for recall              #
#################################################
ax = sns.barplot(
    df_metrics_all_authors,
    x="Method",
    y="Recall (%)",
    hue="Method Type",
    palette=["orangered", "mediumvioletred", "lightseagreen"],
)
ax.tick_params(axis="x", labelrotation=90)
sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)
plt.savefig("recall_all.png", format="png", bbox_inches="tight", dpi=400)
plt.close()
