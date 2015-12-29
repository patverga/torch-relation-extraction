import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors
import sys

# change font family to match math
matplotlib.rc('text', usetex=True)

fontsize = 22
font = {'family' : 'serif',
        'serif' : 'Times Roman',
        'size'   : fontsize}
matplotlib.rc('font', **font)

output_dir = "doc/naacl2016/"

# load in data
xlabels = ["$<3$", "$<5$", "$\geq 5$", "$\geq 10$"]
uschema_f1s = [0.24300966, 0.3140496, 0.20021415, 0.10432721]
lstm_f1s = [0.17596321, 0.28711897, 0.25862822, 0.14799798]

colors=['0.25', '0.75']
bar_width = 0.25

# initialize figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("LSTM + USchema F1: Varying Pattern Length", fontsize=fontsize)
ax1.set_ylabel("F1")
ax1.set_xlabel("Pattern Length")
inds = np.arange(len(uschema_f1s))
ax1.set_xticks(inds + bar_width)
ax1.set_xticklabels(xlabels)
uschema_bar = ax1.bar(inds, uschema_f1s, bar_width, color=colors[0])
lstm_bar = ax1.bar(inds+bar_width, lstm_f1s, bar_width, color=colors[1])

plt.tight_layout()

# add legend
ax1.legend((uschema_bar[0], lstm_bar[0]), ('USchema', 'LSTM'), fontsize=18)

fig1.savefig("%s/f1-vary-pat-length.pdf" % (output_dir), bbox_inches='tight')

plt.show()
