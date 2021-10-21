import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransE_Train_Test


file1 = "..\\..\\WN18\\wordnet-mlj12-train.txt"
file2 = "..\\..\\WN18\\entity2id.txt"
file3 = "..\\..\\WN18\\relation2id.txt"
file4 = "..\\..\\WN18\\wordnet-mlj12-valid.txt"

file5 = "WN18_torch_TransE_entity_50dim_batch4800"
file6 = "WN18_torch_TransE_relation_50dim_batch4800"

file8 = "..\\..\\WN18\\wordnet-mlj12-test.txt"

entity_set, relation_set, triple_list, valid_triple_list = TransE_Train_Test.dataloader(file1, file2, file3, file4)


transE = TransE_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=6.0, norm=1, C=0.25,
                valid_triple_list=valid_triple_list)
transE.data_initialise()
transE.insert_test_data(file5, file6, file8)

transE.test_run(filter=True)