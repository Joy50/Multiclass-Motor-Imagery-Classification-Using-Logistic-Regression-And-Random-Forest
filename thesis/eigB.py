import numpy as np
from scipy import linalg
def gevd(x1,x2,no_pairs):
	print("Value of X1\n",x1)
	x2=np.nan_to_num(x2)
	print(x2)
	ev,vr= linalg.eig(x1,x2,right=True)
	evAbs = np.abs(ev)
	sort_indices = np.argsort(evAbs)
	chosen_indices = np.zeros(2*no_pairs).astype(int)
	chosen_indices[0:no_pairs] = sort_indices[0:no_pairs]
	chosen_indices[no_pairs:2*no_pairs] = sort_indices[-no_pairs:]

	w = vr[:,chosen_indices] # ignore nan entries
	return w
