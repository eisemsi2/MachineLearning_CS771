import numpy as np
from sklearn import linear_model
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	feature = my_map(X_train)
	model = linear_model.LogisticRegression(penalty='l2',tol=1e-4, 
	max_iter = 100000,C= 70.7,random_state = 42)
	model.fit(feature,y_train)
	w = model.coef_[0,:]
	b = model.intercept_[0]
	return w, b


################################
# Non Editable Region Starting #W
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	X = 1-2*X
	n_rows, n_col = X.shape
	for i in range(1,n_col):
		X[:,n_col-i-1] = X[:,n_col-i-1] * X[:,n_col-i]
	
	out_prod = np.einsum('ij,ik->ijk', X, X)
	unique_variables = np.triu_indices(n_col,k=1)
	feat = out_prod[:, unique_variables[0], unique_variables[1]]
	feat = np.concatenate((X,feat),axis=1)
	
	return feat
