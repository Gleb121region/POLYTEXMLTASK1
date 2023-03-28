import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    image = io.imread('krik.jpeg')
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(image)
    print(image.shape)
    data = image / 255.0
    data = data.reshape(199 * 253, 3)

    kmeans = MiniBatchKMeans(4)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    new_image = new_colors.reshape(image.shape)

    kmeans_1 = MiniBatchKMeans(8)
    kmeans_1.fit(data)
    new_colors = kmeans_1.cluster_centers_[kmeans_1.predict(data)]
    new_image_1 = new_colors.reshape(image.shape)

    kmeans_2 = MiniBatchKMeans(16)
    kmeans_2.fit(data)
    new_colors = kmeans_2.cluster_centers_[kmeans_2.predict(data)]
    new_image_2 = new_colors.reshape(image.shape)

    kmeans_3 = MiniBatchKMeans(32)
    kmeans_3.fit(data)
    new_colors = kmeans_3.cluster_centers_[kmeans_3.predict(data)]
    new_image_3 = new_colors.reshape(image.shape)

    fig, ax = plt.subplots(1, 5, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(image)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(new_image)
    ax[1].set_title('4-color Image', size=16)
    ax[2].imshow(new_image_1)
    ax[2].set_title('8-color Image', size=16)
    ax[3].imshow(new_image_2)
    ax[3].set_title('16-color Image', size=16)
    ax[4].imshow(new_image_2)
    ax[4].set_title('32-color Image', size=16)
    plt.show()
