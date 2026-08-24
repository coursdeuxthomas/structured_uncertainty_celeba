import os
os.environ["KAGGLE_USERNAME"] = "votre_pseudo"
os.environ["KAGGLE_KEY"] = "votre_cle"        # les 2 champs de kaggle.json

import kaggle
kaggle.api.dataset_download_files(
    "jessicali9530/celeba-dataset",
    path="celeba_raw",
    unzip=True,
)