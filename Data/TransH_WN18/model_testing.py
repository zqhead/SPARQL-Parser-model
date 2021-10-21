import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransH_Train_Test


file1 = "..\\..\\WN18\\wordnet-mlj12-train.txt"
file2 = "..\\..\\WN18\\entity2id.txt"
file3 = "..\\..\\WN18\\relation2id.txt"
file4 = "..\\..\\WN18\\wordnet-mlj12-valid.txt"

file5 = "WN18_1epoch_TransH_pytorch_entity_50dim_batch4800"
file6 = "WN18_1epoch_TransH_pytorch_norm_relations_50dim_batch4800"
file7 = "WN18_1epoch_TransH_pytorch_hyper_relations_50dim_batch4800"
file8 = "..\\..\\WN18\\wordnet-mlj12-test.txt"

file9 = "" # loss record file

entity_set, relation_set, triple_list, valid_triple_list = TransH_Train_Test.dataloader(file1, file2, file3, file4)


transh = TransH_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.005,
                                            margin=4.0, norm=1, C=0.25, epsilon=1e-5,
                                            valid_triple_list = valid_triple_list)

transh.data_initialise()
transh.insert_data(file5, file6, file7,file8, file9)
transh.test_run(filter=True)