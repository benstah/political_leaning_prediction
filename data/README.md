## Number of Rows for **INTERIM** data ##

- training_set_s: 365389

- training_set_l: 905165

- validation_set: 204

- test_set: 203

### Links for downloading external datasets

To download the POLUSA dataset, you are required to request access to it at: https://zenodo.org/records/3946057

The csv files then need to go into ./external directory.

### Links for downloading/Viewing raw datasets

To process the csv data to the raw data you need to first execute for each csv file individually:

/data_preperation/csv_dump.py

followed by adding the rating column for each raw dataset individually:

/data_preperation/add_rating_column.py

this should leave you with the following structure:

- ./raw/2017_1
- ./raw/2017_2
- ./raw/2018_1
- ./raw/2018_2
- ./raw/2019_1
- ./raw/2019_2 


### Links for downloading/Viewing interim datasets

- ./interim/training_set_s : https://ucloud.univie.ac.at/index.php/s/pmxM5fzeV6BTz6z
- ./interim/validation_set : https://ucloud.univie.ac.at/index.php/s/0DVl1hZuYmhqcfi
- ./interim/test_set :       https://ucloud.univie.ac.at/index.php/s/Mduk47OoG4ngzcR


### Links for downloading/viewing processed datasets

- ./processed/training_set_s : https://ucloud.univie.ac.at/index.php/s/DWF03kSMfopJe7S
- ./processed/validation_set : https://ucloud.univie.ac.at/index.php/s/Wrk7ZWnAIAYhYrO
- ./processed/test_set :       https://ucloud.univie.ac.at/index.php/s/inZfVXCvT31GwWt 

### Links for downloading/viewing processed_vector datasets

- ./processed_vector/training_set_S : https://ucloud.univie.ac.at/index.php/s/2NYpHJfbfPcboE9
- ./processed_vector/validation_set : https://ucloud.univie.ac.at/index.php/s/pqZ94fGGyA8YkjH
- ./processed_vector/test_set       : https://ucloud.univie.ac.at/index.php/s/7zBfjxCfm9z77He
