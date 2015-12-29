import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors
import sys

matplotlib.rc('text', usetex=True)

fontsize = 22
font = {'family' : 'serif',
        'serif' : 'Times Roman',
        'size'   : fontsize}
matplotlib.rc('font', **font)

output_dir = "doc/naacl2016/"

# load in data
data_fname = sys.argv[1]

labels = np.unique(np.loadtxt(data_fname, usecols=[2], dtype='str'))

print labels

data = np.loadtxt(data_fname, converters = {2: lambda y: np.where(labels==y)[0]})

labels = ["LSTM", "USchema"]
colors = ['0.6', '0.25']
width = 4

print data

recall_idx = 0
precision_idx = 1
model_idx = 2

# initialize figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title("LSTM + USchema: Recall vs. Precision", fontsize=fontsize)
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
for i in range(len(labels)):
	indices = np.where(data[:,model_idx] == i)
	ax1.plot(data[indices,recall_idx][0], data[indices,precision_idx][0], label=labels[i], color=colors[i], lw=width)

ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('%.2f')%(y)))

# add legend
ax1.legend()

plt.tight_layout()

fig1.savefig("%s/pr-curve.pdf" % (output_dir), bbox_inches='tight')

plt.show()
