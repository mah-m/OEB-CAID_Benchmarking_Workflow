from __future__ import division, print_function
import os
import json
import sys
from argparse import ArgumentParser
from JSON_templates import JSON_templates
import re

parser = ArgumentParser()

parser.add_argument("-i", "--participant_data",help="the prediction file of the participant", required=True)
parser.add_argument("-com", "--community_name",help="name of benchmarking community", required=True)
parser.add_argument("-c", "--challenges", nargs='+',help="list of challenges of separated by space", required=True)
parser.add_argument("-p", "--participant_name",help="name of the tool used for the prediction", required=True)
# gold_standards_dir must change to public_ref_dir
# parser.add_argument("-r", "--gold_standards_dir",help="directory with the public refrences are.", required=True)
parser.add_argument("-r", "--public_ref_dir",help="directory with the public refrences are. For CAID, public_ref_dir is the same as gold_standards_dir.", required=True)

parser.add_argument("-o", "--output", help="output path where participant JSON file will be written ", required=True)

args = parser.parse_args()


def main(args):
    input_participant = args.participant_data
    community = args.community_name
    challenges = args.challenges
    participant_name = args.participant_name
    # gold_standards_dir = args.gold_standards_dir
    # For CAID, public refrences are the same as Gold standards, because each submitted file by a participant must be checked agains all
    # chosen refrence files challenges . For now, we copy the files in gold standards dir in the public ref dir, and keep the name in order 
    # to be compatible with OEB VRE.
    public_ref_dir=args.public_ref_dir
    out_path = args.output

    
    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
            with open(out_path, mode="a"):
                pass
            print("built the path")
        except OSError as exc:
            print("OS error: {0}".format(exc) +"\nCould not create output path: " + out_path)

    # validate_input_data(input_participant, gold_standards_dir, community, challenges, participant_name, out_path)
    validate_input_data(input_participant, public_ref_dir, community, challenges, participant_name, out_path)


# def validate_input_data(input_participant, gold_standards_dir, community, challenges, participant_name, out_path):
def validate_input_data(input_participant, public_ref_dir, community, challenges, participant_name, out_path):
    validated = False
    pattern = '>DP\w{5}'
    pat = re.compile(pattern)
    for challenge in challenges:
        # challenge_path = os.path.join(gold_standards_dir, challenge+".txt")
        challenge_path = os.path.join(public_ref_dir, challenge+".txt")
        try:
            with open(input_participant) as p, open(challenge_path) as r:
                predictions = p.read()
                predicted_proteins = pat.findall(predictions)
                predicted_proteins_set = set(predicted_proteins)
                reference = r.read()
                reference_proteins = pat.findall(reference)
                reference_proteins_set = set(reference_proteins)
                if not predicted_proteins_set.issubset(reference_proteins_set):
                    print("WARNING: Submitted data file {} contains proteins that are not in the {} reference file! Please check if you have chosen the challenges correctly based on Disorder or Binding types. ".format(input_participant, challenge), file=sys.stderr)
                    validated = False
                proteins = reference_proteins_set & predicted_proteins_set

        except Exception as e:
            print(e)
            sys.exit("ERROR: Submitted data file {} can not be validated against choesn reference file {}!".format(input_participant,challenge))

        with open(input_participant) as p, open(challenge_path) as r:
            predictions = p.readlines()
            predictions = [pred.strip().split() for pred in predictions]

            reference = r.readlines()
            reference = [ref.strip() for ref in reference]

        for protein in proteins:
            ref_idx = reference.index(protein)
            seq_list = reference[ref_idx+1]
            ref_seq_length = len(seq_list)

            pred_idx = predictions.index([protein])

            count = 1
            while ((pred_idx + count + 1) < len(predictions)) and (not len(predictions[pred_idx + count + 1]) == 1):
                count += 1

            pred_seq_length = count

            if not pred_seq_length == ref_seq_length:
                validated = False
                print("WARNING: Protein {protein} in the submitted file has a different length from protein in the {} reference! Please check if you have chosen the challenges correctly based on Disorder or Binding types.".format(protein, challenge), file=sys.stderr)

            else:
                cut = predictions[pred_idx + 1: pred_idx + pred_seq_length + 1]
                cut_seq = [c[1] for c in cut]

                if cut_seq == list(seq_list):
                    validated = True
                else:
                    validated = False
                    print("WARNING: Protein {} in the submitted file has a different sequence from the protein present in the {} reference! Please check if you have chosen the challenges correctly based on Disorder or Binding types.".format(protein, challenge), file=sys.stderr)

            if validated == False:
                sys.exit("ERROR: Submitted file {} does not validate against {} reference file. Please make sure the submitted file can be validated against all of the chosen reference files/challenges.".format(input_participant, challenge))

    data_id = community + ":" + participant_name + "_P"

    output_json = JSON_templates.write_participant_dataset(
        data_id, community, challenges, participant_name, validated)

    with open(out_path, 'w') as f:
        json.dump(output_json, f, sort_keys=True,
                  indent=4, separators=(',', ': '))



if __name__ == "__main__":
    main(args)
