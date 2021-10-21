import numpy
import codecs
import json

def data_parser(file):
    enetity2id = {}
    relation2id = {}
    triples2id = []

    with codecs.open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split('\t')
            if len(item) != 3:
                continue
            if item[0] not in enetity2id.keys():
                enetity2id[item[0]] = len(enetity2id)
            if item[1] not in relation2id.keys():
                relation2id[item[1]] = len(relation2id)
            if item[2] not in enetity2id.keys():
                enetity2id[item[2]] = len(enetity2id)

            triples2id.append([enetity2id[item[0]], relation2id[item[1]], enetity2id[item[2]]])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(enetity2id), len(relation2id), len(triples2id)))

    with codecs.open("..\\WN18\\entity2id.txt", "w") as f1:

        for k in enetity2id:
            f1.write(k + '\t' + str(enetity2id[k]))
            f1.write("\n")

    with codecs.open("..\\WN18\\relation2id.txt", "w") as f2:

        for k in relation2id:
            f2.write(k + '\t' + str(relation2id[k]))
            f2.write("\n")

    with codecs.open("..\\WN18\\train_ids.txt", "w") as f3:

        for k in triples2id:
            f3.write(str(k[0]) + '\t' + str(k[1])+"\t"+str(k[2]))
            f3.write("\n")



if __name__ == '__main__':
    file = '..\\WN18\\wordnet-mlj12-train.txt'
    data_parser(file)