import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

x = np.arange(4)
y = [1.26, 1.445, 1.386, 1.547]
y1 = [0.224, 0.339, 0.318, 0.55]
y2 = [0.488, 0.7, 0.614, 0.905]

bar_width = 0.25
tick_label = ["Q3", "Q4", "Q5", "Q6"]

plt.bar(x, y, bar_width, color="c",align="center", label="TransE", alpha=0.5)
plt.bar(x+bar_width, y1, bar_width, color="b", align="center", label="TransH", alpha=0.5)
plt.bar(x+2*bar_width, y2, bar_width, color="y", align="center", label="TransR", alpha=0.5)

plt.xlabel("Query")
plt.ylabel("Running time(s)")

plt.xticks(x+3*bar_width/2, tick_label)

plt.legend()

plt.show()


x = np.arange(3)
y = [39.806, 41.743, 47.796]
y1 = [41.096, 40.744, 44.318]
y2 = [44.867, 40.24, 51.056]

bar_width = 0.25
tick_label = ["Q7", "Q8", "Q9"]

plt.bar(x, y, bar_width, color="c",align="center", label="TransE", alpha=0.5)
plt.bar(x+bar_width, y1, bar_width, color="b", align="center", label="TransH", alpha=0.5)
plt.bar(x+2*bar_width, y2, bar_width, color="y", align="center", label="TransR", alpha=0.5)

plt.xlabel("Query")
plt.ylabel("Running time(s)")

plt.xticks(x+3*bar_width/2, tick_label)

plt.legend()

plt.show()


x = np.arange(2)
y = [1.572, 365.599]
y1 = [0.465, 378.322]
y2 = [0.736, 383.956]

bar_width = 0.25
tick_label = ["Q1", "Q2"]

plt.bar(x, y, bar_width, color="c",align="center", label="TransE", alpha=0.5)
plt.bar(x+bar_width, y1, bar_width, color="b", align="center", label="TransH", alpha=0.5)
plt.bar(x+2*bar_width, y2, bar_width, color="y", align="center", label="TransR", alpha=0.5)

plt.xlabel("Query")
plt.ylabel("Running time(s)")

plt.xticks(x+3*bar_width/2, tick_label)

plt.legend()

plt.show()