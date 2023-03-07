
import pandas as pd

# drop id set 1
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [8, 15, 31, 38, 47, 57, 66] #idset1
	#ids = [9, 16, 32, 39, 48, 58, 67] # idset 2
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset2_%d.pkl' % rep)

# id set 2
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [9, 16, 32, 39, 48, 58, 67] # idset 2
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset2_%d.pkl' % rep)

# id set 3 
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [10, 17, 33, 40, 49, 59, 68]# idset 3
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset3_%d.pkl' % rep)

# id set 4
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [1, 18, 34, 41, 50, 60, 69] # idset 4
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset4_%d.pkl' % rep)


# id set 5
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [12, 19, 35, 42, 51, 61, 0] # idset 5
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset5_%d.pkl' % rep)

# id set 6
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [13, 20, 36, 43, 52, 62, 1] # idset 6
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset6_%d.pkl' % rep)

# id set 7
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [14, 21, 25, 44, 53, 63, 2] # idset 7
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset7_%d.pkl' % rep)


# id set 8
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [22, 45, 54, 64, 3, 4, 5] # idset 8
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset8_%d.pkl' % rep)

# id set 9
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [23, 24, 55, 65, 6, 7, 56] # idset 9
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset9_%d.pkl' % rep)


#id set 10
for rep in range(1,2):
	merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_%d.pkl' % rep
	df = pd.read_pickle(merged)
	df1 = df
	df2 = df
	ids = [46, 37, 26, 27, 28, 29, 30] # idset 10
	for i,val in enumerate(df1['identity']):
		if val in ids:
			df1 = df1.drop(i)
		else:
			df2 = df2.drop(i)
	df1 = df1.reset_index()
	df2 = df2.reset_index()
	use = list()
	for i,val in enumerate(df1['identity']):
		use.append('Training')
	df1['usage'] = use
	use = list()
	for i,val in enumerate(df2['identity']):
		use.append('PublicTest')
	df2['usage'] = use
	stacked = [df1,df2]
	new_df = pd.concat(stacked, ignore_index=True)
	new_df1 = new_df.drop(columns=['index'])
	new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_idset10_%d.pkl' % rep)

### repeat removing expressions

for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 0:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    #new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_expression_AF_%d.pkl' % rep)
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_AF_%d.pkl' % rep)
    
for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 1:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_AN_%d.pkl' % rep)

for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 2:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_DI_%d.pkl' % rep)


for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 3:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_HA_%d.pkl' % rep)

for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 4:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_NE_%d.pkl' % rep)

for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 5:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_SA_%d.pkl' % rep)

for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet/kdef_features/kdef_features_merged_1.pkl'
    merged = '/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_features_merged_1.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion_num']):
        if val == 6:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion_num']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion_num']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/ieeg/neuralNetworks/expression/resnet_noSoftmax/kdef_features/kdef_expression_SU_%d.pkl' % rep)

    
