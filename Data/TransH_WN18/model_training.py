import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransH_Train_Test



file1 = "..\\..\\WN18\\wordnet-mlj12-train.txt"
file2 = "..\\..\\WN18\\entity2id.txt"
file3 = "..\\..\\WN18\\relation2id.txt"
file4 = "..\\..\\WN18\\wordnet-mlj12-valid.txt"
entity_set, relation_set, triple_list, valid_triple_list = TransH_Train_Test.dataloader(file1, file2, file3, file4)


transH = TransH_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.005, margin=4.0, norm=1, C=0.25, epsilon=1e-5, valid_triple_list = valid_triple_list)
transH.data_initialise()
transH.training_run(epochs=500, batch_size=4800, out_file_title="WN18_1epoch_")