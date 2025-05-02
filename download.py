# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")

# print("Path to dataset files:", path)


import dataset_tools as dtools

dtools.download(dataset="BDD100K: Images 100K", dst_dir="dataset-ninja")
