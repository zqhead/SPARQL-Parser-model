import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransH_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"

file5 = "FB15k_3epoch_TransH_pytorch_entity_200dim_batch9600"
file6 = "FB15k_3epoch_TransH_pytorch_norm_relations_200dim_batch9600"
file7 = "FB15k_3epoch_TransH_pytorch_hyper_relations_200dim_batch9600"
file8 = "..\\..\\FB15k\\freebase_mtr100_mte100-test.txt"
file9 = "" # loss record file

entity_set, relation_set, triple_list, valid_triple_list = TransH_Train_Test.dataloader(file1, file2, file3, file4)



transH = TransH_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=200, lr=0.005, margin=0.25, norm=1, C=1.0, epsilon=1e-5, valid_triple_list = valid_triple_list)
transH.data_initialise()
transH.insert_data(file5, file6, file7, file8, file9)
transH.test_run(filter = False)