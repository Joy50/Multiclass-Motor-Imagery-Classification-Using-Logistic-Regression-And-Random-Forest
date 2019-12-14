from sklearn.decomposition import KernelPCA
def stepwise_kpca(dataset):
    kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True,gamma=10)

    '''n_train_samples,n_train_x,n_train_y = dataset.shape
    train_data_new = dataset.reshape(n_train_samples,n_train_x*n_train_y)'''
    transformed_data = kpca.fit_transform(dataset)

    return transformed_data
