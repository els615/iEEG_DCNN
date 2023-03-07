import pandas as pd
import os
import numpy as np
import scipy.io

### conv1
### if version B 
for rep in range(1,2):
	df_features = pd.read_pickle('/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['conv1'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	order = order.merge(df_layer, on='filepath', how='left')
	#
	# to check merged correctly: df_layer.loc[df_layer['filepath'] == 'AM08AFFL']
	#
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_conv1_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		try:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_conv1_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

#### if version A
for rep in range(1,11):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['conv1'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	for i,file in enumerate(df_layer['filepath']):
		if file[-2:] == 'FL':
			feat1 = df_layer['features'][i]
			feat2name = file[0:-2] + 'FR'
			idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
			if len(idx) >0:
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				#avgFeat = np.average(np.vstack((feat1, feat2)), axis=0)
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer['features'][i] = avgFeat
			else:
				#df_layer['filepath'][i] = file[0:-2] + 'F'
				#df_layer = df_layer.drop([i])  
				continue    
		elif file[-2:] == 'HL':
			feat1 = df_layer['features'][i]
			feat2name = file[0:-2] + 'HR'
			idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
			if len(idx) >0:
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer['features'][i] = avgFeat
			else:
				#df_layer['filepath'][i] = file[0:-2] + 'H'
				#df_layer = df_layer.drop([i])
				continue
		else:
			continue
	order = order.merge(df_layer, on='filepath', how='left')
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_conv1_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		transpose_feat = col_features[i].flatten() # flatten matrix
		transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
		try:
			#transpose_feat = np.array(col_features[i])[np.newaxis]
			#transpose_feat = transpose_feat.flatten() # flatten matrix
			#transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			#transpose_feat = np.array(col_features[i])[np.newaxis]
			#transpose_feat = transpose_feat.flatten() # flatten matrix
			#transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_conv1_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

#### layer 1
### if version B 
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay1'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	order = order.merge(df_layer, on='filepath', how='left')
	#
	# to check merged correctly: df_layer.loc[df_layer['filepath'] == 'AM08AFFL']
	#
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay1_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		if i > 0:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		else:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay1_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})


#### if version A
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay1'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	for i,file in enumerate(df_layer['filepath']):
		if file[-2:] == 'FL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'FR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				#avgFeat = np.average(np.vstack((feat1, feat2)), axis=0)
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer = df_layer.drop([i])      
		elif file[-2:] == 'HL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'HR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer = df_layer.drop([i])
				#continue
		else:
			continue
	order = order.merge(df_layer, on='filepath', how='left')
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay1_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		transpose_feat = col_features[i].flatten() # flatten matrix
		transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
		try:
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay1_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

## layer 2
### if version B 
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay2'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	order = order.merge(df_layer, on='filepath', how='left')
	#
	# to check merged correctly: df_layer.loc[df_layer['filepath'] == 'AM08AFFL']
	#
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay2_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		try:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay2_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

#### if version A
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay2'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	for i,file in enumerate(df_layer['filepath']):
		if file[-2:] == 'FL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'FR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				#avgFeat = np.average(np.vstack((feat1, feat2)), axis=0)
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer = df_layer.drop([i])      
		elif file[-2:] == 'HL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'HR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer = df_layer.drop([i])
				#continue
		else:
			continue
	order = order.merge(df_layer, on='filepath', how='left')
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax//kdef_features/ordered_features/vA_lay2_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		transpose_feat = col_features[i].flatten() # flatten matrix
		transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
		try:
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax//kdef_features/ordered_features/vA_lay2_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

## layer 3
### if version B 
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay3'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	order = order.merge(df_layer, on='filepath', how='left')
	#
	# to check merged correctly: df_layer.loc[df_layer['filepath'] == 'AM08AFFL']
	#
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay3_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		try:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay3_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})
	
#### if version A
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay3'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	for i,file in enumerate(df_layer['filepath']):
		if file[-2:] == 'FL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'FR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				#avgFeat = np.average(np.vstack((feat1, feat2)), axis=0)
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer = df_layer.drop([i])      
		elif file[-2:] == 'HL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'HR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer = df_layer.drop([i])
				#continue
		else:
			continue
	order = order.merge(df_layer, on='filepath', how='left')
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay3_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		transpose_feat = col_features[i].flatten() # flatten matrix
		transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
		try:
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay3_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

## layer 4
### if version B 
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay4'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	order = order.merge(df_layer, on='filepath', how='left')
	#
	# to check merged correctly: df_layer.loc[df_layer['filepath'] == 'AM08AFFL']
	#
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay4_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		try:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			transpose_feat = np.array(col_features[i])[np.newaxis]
			transpose_feat = transpose_feat.flatten() # flatten matrix
			transpose_feat = np.expand_dims(transpose_feat, axis =0)
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vB_lay4_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})
	
#### if version A
for rep in range(1,2):
	df_features = pd.read_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep)
	#order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionB.txt', sep='/n')
	order = pd.read_csv('/mmfs1/data/schwarex/ieeg/versionA.txt', sep='/n')
	for i,file in enumerate(df_features['path']):
		fileShort = os.path.basename(os.path.normpath(file))
		fileShort = fileShort[0:-4]
		df_features['path'][i] = fileShort
	filepath = df_features['path']
	features = df_features['lay4'] #extracted layer 
	cols = list(zip(filepath,features))
	df_layer = pd.DataFrame(cols, columns = ['filepath','features'])
	for i,file in enumerate(df_layer['filepath']):
		if file[-2:] == 'FL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'FR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				#avgFeat = np.average(np.vstack((feat1, feat2)), axis=0)
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'F'
				df_layer = df_layer.drop([i])      
		elif file[-2:] == 'HL':
			feat1 = df_layer['features'][i]
			try:
				feat2name = file[0:-2] + 'HR'
				idx = df_layer[df_layer['filepath'] == feat2name].index.tolist()
				num = idx[0]
				feat2 = df_layer['features'][num]
				avgFeat = (feat1 + feat2) / 2
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer['features'][i] = avgFeat
			except IndexError:
				df_layer['filepath'][i] = file[0:-2] + 'H'
				df_layer = df_layer.drop([i])
				#continue
		else:
			continue
	order = order.merge(df_layer, on='filepath', how='left')
	order.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay4_kdef_ordered_feature_merged_%d.pkl' % rep)
	col_features = pd.Series.to_numpy(order['features'])
	features_flat = None
	#for i, features in enumerate(order['features']):
	for i, features in enumerate(col_features):
		transpose_feat = col_features[i].flatten() # flatten matrix
		transpose_feat = np.expand_dims(transpose_feat, axis =0) #add dimension in x 
		try:
			features_flat = np.concatenate((features_flat, transpose_feat), axis=0)
		except ValueError:
			features_flat = transpose_feat
	scipy.io.savemat('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/ordered_features/vA_lay4_kdef_ordered_feature_merged_%d.mat' % rep, {'struct':features_flat})

