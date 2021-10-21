import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransH_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"


entity_set, relation_set, triple_list, valid_triple_list = TransH_Train_Test.dataloader(file1, file2, file3, file4)


transH =  TransH_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=100, lr=0.005, margin=0.25, norm=1, C=1.0,
                epsilon=1e-5, valid_triple_list=valid_triple_list)
transH.data_initialise()

transH.training_run(epochs=50, batch_size=4800, out_file_title="FB15k_200epoch_")
