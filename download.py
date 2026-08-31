import os
os.environ["KAGGLE_USERNAME"] = "votre_pseudo"
os.environ["KAGGLE_KEY"] = "votre_cle"        # the 2 fields in kaggle.json

import kaggle
kaggle.api.dataset_download_files(
    "jessicali9530/celeba-dataset",
    path="celeba_raw",
    unzip=True,
)