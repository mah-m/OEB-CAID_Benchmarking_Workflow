from __future__ import division
import io
from operator import index
import os
import pandas as pd
import math
import json
from argparse import ArgumentParser
from JSON_templates import JSON_templates

def main(args):
    input_participant=args.participant_data
    community=args.community_name
    challenges=args.challenges
    participant_name=args.participant_name
    gold_standards_dir=args.metrics_ref
    out_path=args.output

    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
            with open(out_path,mode="a"): pass
            print("built the path")
        except OSError as exc:
            print("OS error: {0}".format(exc) + "\nCould not create output path: " + out_path)
    compute_metrics(input_participant, gold_standards_dir, challenges, participant_name, community, out_path)


def compute_metrics(input_participant, gold_standards_dir, challenges, participant_name, community, out_path):
    participant_data=pd.read_csv(input_participant,header=0,sep='\t')

    # TODO: Metrics Computation part must be completed according to the 
    predictions = participant_data.loc[:,'disordered'].values

    ALL_ASSESSMENTS=[]
    for challenge in challenges:
        metrics_data=pd.read_csv(os.path.join(gold_standards_dir,challenge+".txt"),index_col=0,sep='\t',comment="#",header=0)
        print(metrics_data.head())
        gold_standard = metrics_data.iloc[:,2].values

        # TODO : TPR or RECALL
        TP = sum(predictions & gold_standard)
        FN = sum(gold_standard) - TP
        TPR = TP / (TP + FN)
        print('recall is ',TPR)

        # TODO : PRECISION
        FP = sum(predictions) - TP
        precision=TP / (TP + FP)
        print('precision is ',precision)

        # TODO : ACCURACY
        accuracy = sum(gold_standard & predictions) / len(predictions)
        print('accuracy is ',accuracy)

        assessment_data= {'toolname':participant_name,'x': TPR,'y':accuracy,'e':0,'challenge':challenge}
        data_id_1 = community + ":" + challenge + "_TPR_" + participant_name + "_A"
        std_error= 0
        assessment_TPR = JSON_templates.write_assessment_dataset(data_id_1, community, challenge, participant_name, "TPR", TPR, std_error)

        data_id_2 = community + ":" + challenge + "_precision_" + participant_name + "_A"
        assessment_precision = JSON_templates.write_assessment_dataset(data_id_2, community, challenge, participant_name, "precision", precision, std_error)

        data_id_3 = community + ":" + challenge + "_accuracy_" + participant_name + "_A"
        assessment_accuracy = JSON_templates.write_assessment_dataset(data_id_3, community, challenge, participant_name, "accuracy", accuracy, std_error)
        ALL_ASSESSMENTS.extend([assessment_TPR,assessment_precision,assessment_accuracy])

    with io.open(out_path,mode='w', encoding="utf-8") as f:
        jdata = json.dumps(ALL_ASSESSMENTS, sort_keys=True, indent=4, separators=(',', ': '))
        f.write(jdata)    

if __name__=='__main__':
    parser = ArgumentParser()
    # TODO: make the help phrases better
    parser.add_argument("-i","--participant_data",help="list of the predictions",required=True)
    parser.add_argument("-com","--community_name",help="name of benchmarking community",required=True)
    # TODO: the example is cancer type, but here we have binding and disorder. make a united name 
    # for them. for now it is DISORDERED.
    parser.add_argument("-c","--challenges", nargs='+',help="list of challenges, disorders or bindings, separated by space",required=True)
    parser.add_argument("-p","--participant_name",help="name of the tool used for the prediction",required=True)
    parser.add_argument("-m", "--metrics_ref", help="dir that contains metrics reference datasets for all disorder types", required=True)
    parser.add_argument("-o","--output",help="output path where assessment JSON files will be written ",required=True)
    args=parser.parse_args()

    main(args)