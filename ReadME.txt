This is the Program of SPARQL Parser Model.

If you want to run this program, you need to install Pytorch and rdflib package Firstly.

This Model has 4 modules.

The Embedding Processor is in "module_Training_Testing" folder.
The Data Preprocessor is in the "Text2RDF.py" file.
THe Query Parser and Scoring and Ranking module are in the "SPARQLPARSER.py" file.

As the embedding data files are big, this project can't contain them and you need to training them firstly.
Or you can follow below steps.

Embedding Processor:
Firstly, you need to use the file "model_training.py" in "Data" folder to train embedding data for each model and dataset
And then you can use the run the file "model_testing.py" to test the proformance of the model.

Data Preprocessor:
Firstly, you need to run the files "WN18_process.py" or "FB15k_process.py" in the "Data_preprocess" folder to generate the RDF Graph
and then run the file "preprocess.py" in each folder of "DATA" folder to translate the entities and relations to the URL address.


Query Parser and Scoring and Ranking
After Training Embedding data and preprocessing the dataset, you can run the "SPARQLPARSER.py" to get the SPARQL Queries answers.
There are some examples in the "Example" folder, including the simple query test or the conjunctive queries which are described in my paper.



