import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransR_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"

file5 = "FB15k_torch_TransR_pytorch_entity_50dim_batch9600"
file6 = "FB15k_torch_TransR_pytorch_reltion_50dim_batch9600"
file7 = "FB15k_torch_TransR_pytorch_rel_matrix_50_50dim_batch9600"
file8 = "..\\..\\FB15k\\freebase_mtr100_mte100-test.txt"


entity_set, relation_set, triple_list, valid_triple_list = TransR_Train_Test.dataloader(file1, file2, file3, file4)



transR = TransR_Train_Test.TransR_Training_Testing(entity_set, relation_set, triple_list, ent_dim=50, rel_dim=50, lr=0.00001, margin=6.0,
                         norm=1, C=0.25, valid_triples=valid_triple_list)
transR.data_initialise()
transR.insert_test_data(file5, file6, file7, file8)
transR.test_run(filter = True)