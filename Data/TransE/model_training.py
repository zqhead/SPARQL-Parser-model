import sys
sys.path.append("..")
import Text2RDF as T2R
from module_Traning_Testing import TransE_Train_Test


file1 = "..\\..\\FB15k\\freebase_mtr100_mte100-train.txt"
file2 = "..\\..\\FB15k\\entity2id.txt"
file3 = "..\\..\\FB15k\\relation2id.txt"
file4 = "..\\..\\FB15k\\freebase_mtr100_mte100-valid.txt"


entity_set, relation_set, triple_list, valid_triple_list = TransE_Train_Test.dataloader(file1, file2, file3, file4)

transE = TransE_Train_Test.Training_Testing(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.001, margin=4.0, norm=1, C = 0.25, valid_triple_list=valid_triple_list)
transE.data_initialise()
transE.training_run(epochs=500, batch_size=9600, out_file_title="KB15k_torch_")