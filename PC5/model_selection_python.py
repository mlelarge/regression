import statsmodels.api as sm
import numpy as np
import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LinearRegression



def AIC(data,model,model_type):
	if model_type=='linear':
		return len(data)* np.log(model.ssr/len(data)) + 2 * (model.df_model+1)
	else :
		return model.aic

def Cp(data,model,sigma2):
	return model.ssr/sigma2 - (len(data) - 2.*model.df_model- 1)

def BIC(data,model):
	return np.log(model.ssr/model.centered_tss) * len(data) + (model.df_model+1) * np.log(len(data))

def forwardSelection(X, y, model_type ="linear",elimination_criteria = "aic", varchar_process = "dummy_dropfirst",verbose=False):
	X = __varcharProcessing__(X,varchar_process = varchar_process)
	return __forwardSelectionRaw__(X, y, model_type = model_type,elimination_criteria = elimination_criteria,verbose=False)

def backwardSelection(X, y, model_type ="linear",elimination_criteria = "aic", varchar_process = "dummy_dropfirst",verbose=False):

	X = __varcharProcessing__(X,varchar_process = varchar_process)
	return __backwardSelectionRaw__(X, y, model_type = model_type,elimination_criteria = elimination_criteria,verbose=False )


def bothSelection(X, y, model_type ="linear",elimination_criteria = "aic",start='full', varchar_process = "dummy_dropfirst",verbose=False):

	X = __varcharProcessing__(X,varchar_process = varchar_process)
	return __bothSelectionRaw__(X, y, model_type = model_type,elimination_criteria = elimination_criteria,start=start,verbose=False)


def __varcharProcessing__(X, varchar_process = "dummy_dropfirst"):

	dtypes = X.dtypes
	if varchar_process == "drop":
		X = X.drop(columns = dtypes[dtypes == np.object].index.tolist())
		print("Character Variables (Dropped):", dtypes[dtypes == np.object].index.tolist())
	elif varchar_process == "dummy":
		X = pd.get_dummies(X,drop_first=False)
		print("Character Variables (Dummies Generated):", dtypes[dtypes == np.object].index.tolist())
	elif varchar_process == "dummy_dropfirst":
		X = pd.get_dummies(X,drop_first=True)
		print("Character Variables (Dummies Generated, First Dummies Dropped):", dtypes[dtypes == np.object].index.tolist())
	else:
		X = pd.get_dummies(X,drop_first=True)
		print("Character Variables (Dummies Generated, First Dummies Dropped):", dtypes[dtypes == np.object].index.tolist())

	# X["const"] = 1
	cols = X.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	X = X[cols]

	return X

def __forwardSelectionRaw__(X, y, model_type ="linear",elimination_criteria = "aic",verbose=False):

	cols = X.columns.tolist()

	def regressor(y,X, model_type=model_type):
		if model_type == "linear":
			regressor = sm.OLS(y, X).fit()
		elif model_type == 'logistic':
			y = pd.get_dummies(y,drop_first=True)
			y_array = np.asarray(y)
			X_array = np.asarray(X)
			regressor = sm.GLM(y_array, X_array,family=sm.families.Binomial()).fit()
		return regressor

	## Begin from simple model with only intercept
	selected_cols = ["const"]
	other_cols = cols.copy()
	other_cols.remove("const")

	model = regressor(y, X[selected_cols])

	if elimination_criteria == "aic":
		criteria = AIC(X,model,model_type)


	for i in range(X.shape[1]):
		aicvals = pd.DataFrame(columns = ["Cols","aic"])
		for j in other_cols:
			model = regressor(y, X[selected_cols+[j]])
			aicvals = aicvals.append(pd.DataFrame([[j, AIC(X,model,model_type)]],columns = ["Cols","aic"]),ignore_index=True)

		aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)
		if verbose :
			print(aicvals)
		if aicvals.shape[0] > 0:

			if  elimination_criteria == "aic":
				new_criteria = aicvals["aic"][0]
				if new_criteria < criteria:
					print("Entered :", aicvals["Cols"][0], "\tAIC :", aicvals["aic"][0])
					selected_cols.append(aicvals["Cols"][0])
					other_cols.remove(aicvals["Cols"][0])
					criteria = new_criteria
				else:
					print("break : Criteria")
					break

	model = regressor(y, X[selected_cols])

	print(model.summary())
	print("AIC: "+str(AIC(X,model,model_type)))
	print("Final Variables:", selected_cols)

	return model

def __backwardSelectionRaw__(X, y, model_type ="linear",elimination_criteria = "aic",verbose=False):


	cols = X.columns.tolist()

	def regressor(y,X, model_type=model_type):
		if model_type =="linear":
			regressor = sm.OLS(y, X).fit()
		elif model_type == 'logistic':
			y = pd.get_dummies(y,drop_first=True)
			y_array = np.asarray(y)
			X_array = np.asarray(X)
			regressor = sm.GLM(y_array, X_array,family=sm.families.Binomial()).fit()
		return regressor

	model = regressor(y,X)
	criteria = AIC(X,model,model_type)

	for i in range(X.shape[1]):

		cols = X.columns.tolist()
		aicvals = pd.DataFrame(columns = ["Cols","aic"])
		if len(cols)==1:
			print("break : Only one variable left")
			break
		else :
			for j in cols:
				temp_cols = cols.copy()
				temp_cols.remove(j)
				model = regressor(y, X[temp_cols])
				aicvals = aicvals.append(pd.DataFrame([[j, AIC(X,model,model_type)]],columns = ["Cols","aic"]),ignore_index=True)

			aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)
			if verbose :
				print(aicvals)
			if aicvals.shape[0] > 0:
				new_criteria = aicvals["aic"][0]
				if new_criteria < criteria:
					print("Eliminated :" ,aicvals["Cols"][0])
					del X[aicvals["Cols"][0]]
					criteria = new_criteria
				else:
					print("break : Criteria")
					break

	model = regressor(y,X)
	print(str(model.summary())+"\nAIC: "+ str(AIC(X,model,model_type)))
	print("Final Variables:", cols)

	return model

def __bothSelectionRaw__(X, y, model_type ="linear",elimination_criteria = "aic",start='full',verbose=False):


	cols = X.columns.tolist()

	if start=='full':
		removed_cols = []
		selected_cols = cols.copy()
	else :
		selected_cols = ["const"]
		removed_cols = cols.copy()
		removed_cols.remove("const")


	def regressor(y,X, model_type=model_type):
		if model_type =="linear":
			regressor = sm.OLS(y, X).fit()
		elif model_type == 'logistic':
			y = pd.get_dummies(y,drop_first=True)
			y_array = np.asarray(y)
			X_array = np.asarray(X)
			regressor = sm.GLM(y_array, X_array,family=sm.families.Binomial()).fit()
		return regressor

	model = regressor(y,X[selected_cols])
	criteria = AIC(X,model,model_type)

	while True :
		aicvals = pd.DataFrame(columns = ["Cols","aic",'way'])
		if len(selected_cols)==1:
			continue
		else :
			for j in selected_cols:
				temp_cols = selected_cols.copy()
				temp_cols.remove(j)
				model = regressor(y, X[temp_cols])
				aicvals = aicvals.append(pd.DataFrame([[j, AIC(X,model,model_type),'delete']],columns = ["Cols","aic",'way']),ignore_index=True)

		for j in removed_cols:
			model = regressor(y, X[selected_cols+[j]])
			aicvals = aicvals.append(pd.DataFrame([[j, AIC(X,model,model_type),'add']],columns = ["Cols","aic",'way']),ignore_index=True)

		aicvals = aicvals.sort_values(by = ["aic"]).reset_index(drop=True)
		if verbose :
			print(aicvals)
		if aicvals.shape[0] > 0:
			new_criteria = aicvals["aic"][0]
			if new_criteria < criteria:
				if aicvals["way"][0]=='delete':
					print("Eliminated :" ,aicvals["Cols"][0],"\tAIC :", aicvals["aic"][0])
					criteria = new_criteria
					removed_cols.append(aicvals["Cols"][0])
					selected_cols.remove(aicvals["Cols"][0])
				elif aicvals["way"][0]=='add':
					print("Entered :", aicvals["Cols"][0], "\tAIC :", aicvals["aic"][0])
					selected_cols.append(aicvals["Cols"][0])
					removed_cols.remove(aicvals["Cols"][0])
					criteria = new_criteria
			else:
				print("break : Criteria")
				break

	model = regressor(y,X[selected_cols])
	print(str(model.summary())+"\nAIC: "+ str(AIC(X,model,model_type)))
	print("Final Variables:", selected_cols)

	return model


def exhaustivesearch_selectionmodel(X,y,vmin=1,vmax=10):

	'''
	X : Dataframe of explanatory variables, without intercept column
	y : Dataframe of output variable
	Function to compute exhaustive search for linear regression y ~X  : test all models with p features from X between vmin and vmax.
	For each size of models select the best one.
	Then compute R2,adj R2, Cp and BIC for these models.
	---------
	Return these criteria in a DataFrame.

	'''
	lm = LinearRegression(fit_intercept=True)
	efs1 = EFS(lm,min_features=1,
max_features=vmax,
scoring='neg_mean_squared_error',
print_progress=True,
cv=False)

	efs1 = efs1.fit(X, y)

	best_idxs_all = []
	for k in range(1,vmax):
		best_score = -np.infty
		best_idx = 0
		for i in efs1.subsets_:
			if (len(efs1.subsets_[i]['feature_idx'])) == k:
				if efs1.subsets_[i]['avg_score'] > best_score:
					best_score = efs1.subsets_[i]['avg_score']
					best_idx = i

		best_idxs_all.append(best_idx)


	df_subsets = pd.DataFrame(index=best_idxs_all,columns=['Variables','R2','R2_adj','Cp','BIC'])

	X = sm.add_constant(X)
	full_model = sm.OLS(y,X).fit()
	sigma2 = (full_model.ssr)/(len(X)-full_model.df_model-1)


	for index in best_idxs_all:
		variables = np.asarray(efs1.subsets_[index]['feature_names'])
		df_subsets.loc[index,'Variables'] = variables
		variables =  np.concatenate([['const'],variables])
		model = sm.OLS(y,X[variables]).fit()
		df_subsets.loc[index,'R2'] = model.rsquared
		df_subsets.loc[index,'R2_adj'] = model.rsquared_adj
		df_subsets.loc[index,'BIC'] = BIC(X,model)
		df_subsets.loc[index,'Cp'] = Cp(X,model,sigma2)


	return df_subsets
