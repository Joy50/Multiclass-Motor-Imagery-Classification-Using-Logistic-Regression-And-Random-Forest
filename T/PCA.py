from sklearn.decomposition import PCA
def applyPCA(ndimention,dataset):
    pca=PCA(n_components=ndimention)
    pca.fit(dataset)
    r_pca = pca.transform(dataset)
    return r_pca
