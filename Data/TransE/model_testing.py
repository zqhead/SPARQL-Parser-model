import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransE_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"

file5 = "KB15k_torch_TransE_entity_50dim_batch9600"
file6 = "KB15k_torch_TransE_relation_50dim_batch9600"

file8 = "..\\..\\FB15k\\freebase_mtr100_mte100-test.txt"

entity_set, relation_set, triple_list, valid_triple_list = TransE_Train_Test.dataloader(file1, file2, file3, file4)

transE = TransE_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.001, margin=4.0, norm=1, C = 0.25, valid_triple_list=valid_triple_list)
transE.data_initialise()
transE.insert_test_data(file5, file6, file8)

transE.test_run(filter=False)