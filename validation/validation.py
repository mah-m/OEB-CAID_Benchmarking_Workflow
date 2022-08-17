from __future__ import division, print_function
import pandas as pd
import os,json
import sys
from argparse import ArgumentParser
from JSON_templates import JSON_templates

parser=ArgumentParser()
# TODO: make the help phrases better
parser.add_argument("-i","--participant_data",help="list of the predictions",required=True)
parser.add_argument("-com","--community_name",help="name of benchmarking community",required=True)
parser.add_argument("-c","--challenges", nargs='+',help="list of challenges of disorder or binding, separated by space",required=True)
parser.add_argument("-p","--participant_name",help="name of the tool used for the prediction",required=True)
parser.add_argument("-r","--public_ref_dir",help="directory with the list of disordered residues",required=True)
parser.add_argument("-o","--output",help="output path where participant JSON file will be written ",required=True)

args=parser.parse_args()

def main(args):
    input_participant=args.participant_data
    community=args.community_name
    challenges=args.challenges
    participant_name=args.participant_name
    public_ref_dir=args.public_ref_dir
    out_path=args.output

    # assuring that the output path exists
    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
            with open(out_path,mode="a"): pass
            print("built the path")
        except OSError as exc:
            print("OS error: {0}".format(exc) + "\nCould not create output path: " + out_path)

    validate_input_data(input_participant,public_ref_dir,community,challenges,participant_name,out_path)

def validate_input_data(input_participant,public_ref_dir,community,challenges,participant_name,out_path):
    try:
        participant_data=pd.read_csv(input_participant,header=0,sep='\t')
    except:
        sys.exit("ERROR: Submitted data file {} is not in a valid format!".format(input_participant))
    
    
    predicted_proteins = list(participant_data.iloc[:,0].values)
    data_fields=['protein','resn','res','propensity','disordered']
    submitted_fields=list(participant_data.columns.values)

    validated=True
    # for public_ref_rel in os.listdir(public_ref_dir):
    #     public_ref = os.path.join(public_ref_dir,public_ref_rel)
    #     if os.path.isfile(public_ref):
    #         try:
    #             public_ref_data = pd.read_csv(public_ref, sep='\t',
    #                                               comment="#", header=0)
    #             disordered_proteins = list(public_ref_data.iloc[:, 0].values)
                
    #             ## validate the fields of the submitted data and if the predicted genes are in the mutations file
    #             if data_fields == submitted_fields and (set(predicted_proteins) < set(disordered_proteins)) == True:
    #                 validated = True
    #             else:
    #                 print("WARNING: Submitted data does not validate against "+public_ref,file=sys.stderr)
    #                 validated = False
    #         except:
    #             print("PARTIAL ERROR: Unable to properly process "+public_ref,file=sys.stderr)
    #             import traceback
    #             traceback.print_exc()
    
    
    # TODO: data_id can be modified to have BINDING or DISORDER in it. 
    data_id = community + ":" + participant_name + "_P"
    print('data id is ',data_id)
    output_json = JSON_templates.write_participant_dataset(data_id, community, challenges, participant_name, validated)

    # print file

    with open(out_path , 'w') as f:
        json.dump(output_json, f, sort_keys=True, indent=4, separators=(',', ': '))
        print(output_json)
        print('dumped the data')

    if validated == True:

        sys.exit(0)
    else:
        sys.exit("ERROR: Submitted data does not validate against any reference data! Please check " + out_path)

if __name__=="__main__":
    main(args)

