import pandas as pd
for rep in range(1,2):
    merged = '/mmfs1/data/schwarex/neuralNetworks/datasets/KDEF_3views_pixels.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 0:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df1 = new_df.drop(columns=['index'])
    new_df1.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_AF.pkl')
    ######  HL
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 1:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df2 = new_df.drop(columns=['index'])
    new_df2.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_AN.pkl')
    ######  HR
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 2:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df3 = new_df.drop(columns=['index'])
    #new_df3.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_DI_%d.pkl' % rep)
    new_df3.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_DI.pkl')
    ######  HR
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 3:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df4 = new_df.drop(columns=['index'])
    #new_df4.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_HA_%d.pkl' % rep)
    new_df4.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_HA.pkl')
    ######  HR
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 4:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df5 = new_df.drop(columns=['index'])
    #new_df5.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_NE_%d.pkl' % rep)
    new_df5.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_NE.pkl')
    ######  HR
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 5:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df6 = new_df.drop(columns=['index'])
    #new_df6.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_SA_%d.pkl' % rep)
    new_df6.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_SA.pkl')
    ######  HR
    df1 = df
    df2 = df
    for i,val in enumerate(df1['emotion']):
        if val == 6:
            df1 = df1.drop(i)
        else:
            df2 = df2.drop(i)
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    use = list()
    for i,val in enumerate(df1['emotion']):
        use.append('Training')
    df1['usage'] = use
    use = list()
    for i,val in enumerate(df2['emotion']):
        use.append('PublicTest')
    df2['usage'] = use
    stacked = [df1,df2]
    new_df = pd.concat(stacked, ignore_index=True)
    new_df7 = new_df.drop(columns=['index'])
    #new_df7.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_SU_%d.pkl' % rep)
    new_df6.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_SU.pkl')

##### identity DROPS
# set1 
for rep in range(1,2):
    #merged = '/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_%d.pkl' % rep
    merged = '/mmfs1/data/schwarex/neuralNetworks/datasets/KDEF_3views_pixels.pkl'
    df = pd.read_pickle(merged)
    df1 = df
    df2 = df
    ids = [8, 15, 31, 38, 47, 57, 66]
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
    new_df1.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset1.pkl')
    ### idset2
    df1 = df
    df2 = df
    ids = [9, 16, 32, 39, 48, 58, 67]
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
    new_df2 = new_df.drop(columns=['index'])
    #new_df2.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset2_%d.pkl' % rep)
    new_df2.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset2.pkl')
    ### idset3
    df1 = df
    df2 = df
    ids = [10, 17, 33, 40, 49, 59, 68]
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
    new_df3 = new_df.drop(columns=['index'])
    #new_df3.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset3_%d.pkl' % rep)
    new_df3.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset3.pkl')
    ### idset4
    df1 = df
    df2 = df
    ids = [11, 18, 34, 41, 50, 60, 69]
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
    new_df4 = new_df.drop(columns=['index'])
    #new_df4.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset4_%d.pkl' % rep)
    new_df4.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset4.pkl')
    ### idset5
    df1 = df
    df2 = df
    ids = [12, 19, 35, 42, 51, 61, 0]
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
    new_df5 = new_df.drop(columns=['index'])
    #new_df5.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset5_%d.pkl' % rep)
    new_df5.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset5.pkl')
    ### idset6
    df1 = df
    df2 = df
    ids = [13, 20, 36, 43, 52, 62, 1]
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
    new_df6 = new_df.drop(columns=['index'])
    #new_df6.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset6_%d.pkl' % rep)
    new_df6.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset6.pkl')
    ### idset7
    df1 = df
    df2 = df
    ids = [14, 21, 25, 44, 53, 63, 2]
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
    new_df7 = new_df.drop(columns=['index'])
    #new_df7.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset7_%d.pkl' % rep)
    new_df7.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset7.pkl')
    ### idset8
    df1 = df
    df2 = df
    ids = [22, 45, 54, 64, 3, 4, 5]
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
    new_df8 = new_df.drop(columns=['index'])
    #new_df8.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset8_%d.pkl' % rep)
    new_df8.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset8.pkl')
    ### idset9
    df1 = df
    df2 = df
    ids = [23, 24, 55, 65, 6, 7, 56]
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
    new_df9 = new_df.drop(columns=['index'])
    #new_df9.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset9_%d.pkl' % rep)
    new_df9.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset9.pkl')
    ### idset10
    df1 = df
    df2 = df
    ids = [46, 37, 26, 27, 28, 29, 30]
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
    new_df10 = new_df.drop(columns=['index'])
    #new_df10.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/identity/kdef/kdef_features_merged_idset10_%d.pkl' % rep)
    new_df9.to_pickle('/mmfs1/data/schwarex/neuralNetworks/densenet/retrains/pixel_linear/kdef_pixel_idset10.pkl')
    
