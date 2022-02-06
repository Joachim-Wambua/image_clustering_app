from fileinput import filename
import os
import pickle
from tokenize import group
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def list_images(path):
    path = r"/var/www/html/cluster_app/static/upload"

    # Open img/ directrory
    os.chdir(path)

    # List to hold images
    prod_images = []

    # Scans directory while collecting image files
    with os.scandir(path) as files:
        # Loop through files
        for file in files:
            if file.name.endswith('.jpg'):
                # Add to list
                prod_images.append(file.name)
        return prod_images

def image_feature_extractor(image, model):
  # Load image as array of size 224x224
  imge = load_img(image, target_size=(224,224))

  # Convert image to numpy array
  imge = np.array(imge)

  # reshape data for model reshape
  reshaped_img = imge.reshape(1, 224, 224, 3)

  # Prepare Image for the model
  imgx = preprocess_input(reshaped_img)

  # Get Feature Vector
  features = model.predict(imgx, use_multiprocessing=True)
  return features

# Visualise Clusters
def visualise_cluster_opt(cluster,groups):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:10]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')



@app.route("/")
# Render the home page
def home():
    return render_template('index.html')

@app.route("/submit", methods=['GET','POST'])
def predict():
    if request.method == "POST":
        # image_path = ''
        prod_images = []
        data = {}
        files = request.files.getlist('file')
        for file in files:
            UPLOAD_FOLDER = 'static/upload/'
            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
            # files.append(image_path)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            
        img_arr = list_images('/var/www/html/cluster_app/static/upload')
        n=len(img_arr)
        image_model = VGG16(weights='imagenet')
        image_model = Model(inputs = image_model.inputs, outputs = image_model.layers[-2].output)

        for pic in img_arr:
            # Extract Features
            features = image_feature_extractor(pic, image_model)
            data[pic] = features
        # Get list of image filenames
        filenames = np.array(list(data.keys()))

        # get a list of just the features
        features = np.array(list(data.values()))
        features = features.reshape(-1,4096)

        # Dimensionality Reduction using PCA
        pca = PCA(n_components=n, random_state=22)
        pca.fit(features)
        x = pca.transform(features)
        # the scaler object (model)
        scaler = StandardScaler()
        # fit and transform the data
        scaled_data = scaler.fit_transform(x) 
        # file_name = '/var/www/html/cluster_app/image_clustering_model.pkl'
        # cluster_model = pickle.load(open(file_name, 'rb'))
        opt_kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10, max_iter=300, random_state=42)
        opt_kmeans.fit(scaled_data)
        # cluster_model.fit(scaled_data)
        # cluster_model.predict(scaled_data)
        # holds the cluster id and the images { id: [images] }
        groups = {}
        for file, cluster in zip(filenames,opt_kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)
        
        # result1 = visualise_cluster_opt(0, groups)
        grp1 = groups[0][:10]
        grp2 = groups[1][:10]

        return render_template("index.html", cluster1 = "Cluster 0: " + ", ".join(grp1), cluster2 = "Cluster 1: " + "".join(grp2))

if __name__ == '__main__':
    app.run(debug=True)