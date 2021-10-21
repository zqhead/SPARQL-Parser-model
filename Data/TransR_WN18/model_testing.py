import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransR_Train_Test


file1 = "..\\..\\WN18\\wordnet-mlj12-train.txt"
file2 = "..\\..\\WN18\\entity2id.txt"
file3 = "..\\..\\WN18\\relation2id.txt"
file4 = "..\\..\\WN18\\wordnet-mlj12-valid.txt"

file5 = "WN18_1torch_TransR_pytorch_entity_50dim_batch4800"
file6 = "WN18_1torch_TransR_pytorch_reltion_50dim_batch4800"
file7 = "WN18_1torch_TransR_pytorch_rel_matrix_50_50dim_batch4800"
file8 = "..\\..\\WN18\\wordnet-mlj12-test.txt"


entity_set, relation_set, triple_list, valid_triple_list = TransR_Train_Test.dataloader(file1, file2, file3, file4)


transR = TransR_Train_Test.TransR_Training_Testing(entity_set, relation_set, triple_list, ent_dim=50, rel_dim = 50,
                                                   lr=0.001, margin=4.0, norm=1, C=1.0, valid_triples=valid_triple_list)

transR.data_initialise()
transR.insert_test_data(file5, file6, file7,file8)
transR.test_run(filter=True)