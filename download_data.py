import gdown

url = 'https://drive.google.com/drive/folders/12zOqHG6C2o9qEFscZvW1b-xRjhXiy34K?usp=sharing'

gdown.download_folder(url, quiet=False)