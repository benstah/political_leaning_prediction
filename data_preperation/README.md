# Structure and files of this directory

This directory uses 3 files (2017_1, 2018_2, 2019_2) which are taken out of the polusa dataset and have been manipulated in a way to use them for further processing.They are being found in ```data/raw/*```

# step 1
All 3 files are originating from csv's and have been transformed to a panda structure and saved in a compressed format through joblib. This has been done by the '''csv_dump.py''' script.

# step 2
An extra column has been added to the files, which is called 'rating'. This column is being used for identifying the self rated political leaning and has been added by making use of the script '''add_rating_column.py'''.

# step 3
A tool was build with the help of tkinter which helps to read articles and rate them by the identified political leaning. One row of the dataset is always displayed with the index of the row, the header of the article, the body of the article and the self rated political leaning. 
The scripts of '''read_articles.py''' (2017_1), '''read_articles_2018.py''' (2018_2) and '''read_articles_2019.py''' (2019_2) have been used for this porpuse.

# step 4
Lastly the 3 prepared files are being used to create the needed data sets (create_data_sets.py):

  - Training set (weakly labeled data. Only prelabeled data)
  - Validation set (undefined data that will be automatically labeled to a later point)
  - 'Gold' validation set (approx. 200 articles that have a self identified label)
  - Test set (approx. 200 articles for testing)




# Public links for datasets:

- 2017_1 : https://ucloud.univie.ac.at/index.php/s/bbbhsqsRYWraYd0
- 2017_2 : https://ucloud.univie.ac.at/index.php/s/xooDNuoLZoOmaDr
- 2018_1 : https://ucloud.univie.ac.at/index.php/s/8u7LC7kgaWbHvv0
- 2018_2 : https://ucloud.univie.ac.at/index.php/s/s3rAAiuonXAyHGf
- 2019_1 : https://ucloud.univie.ac.at/index.php/s/Pd1VLIKlkpTjYv4
- 2019_2 : https://ucloud.univie.ac.at/index.php/s/7naiqlJSRRK6ur7
