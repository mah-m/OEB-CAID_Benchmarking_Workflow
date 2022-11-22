from __future__ import division
# import requests
import json
import os
from argparse import ArgumentParser
import datetime
from assessment_chart import assessment_chart

DEFAULT_eventMark = '2022-08-16'
DEFAULT_OEB_API = "https://dev-openebench.bsc.es/api/scientific/graphql"
DEFAULT_eventMark_id = "OEBCAID"

def main(args):

    # input parameters
    data_dir = args.benchmark_data 
    participant_path = args.participant_data  # assessment results
    output_dir = args.output
    # offline = args.offline
    
    # Assuring the output directory does exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # read participant metrics
    participant_data = read_participant_data(participant_path)

    # if offline is None:
    #     response = query_OEB_DB(DEFAULT_eventMark_id)
    #     getOEBAggregations(response, data_dir)
    generate_manifest(data_dir, output_dir, participant_data)

##get existing aggregation datasets for that challenges
# def query_OEB_DB(bench_event_id):
#     json_query = {'query': """query AggregationQuery($bench_event_id: String) {
#     getChallenges(challengeFilters: {benchmarking_event_id: $bench_event_id}) {
#         _id
#         acronym
#         metrics_categories{
#           metrics {
#             metrics_id
#             orig_id
#           }
#         }
#         datasets(datasetFilters: {type: "aggregation"}) {
#                 _id
#                 _schema
#                 orig_id
#                 community_ids
#                 challenge_ids
#                 datalink {
#                     inline_data
#                 }
#         }
#     }
# }""",
#                 'variables': {
#                     'bench_event_id': bench_event_id
#                 }
#             }
#     try:
#         url = DEFAULT_OEB_API
#         # get challenges and input datasets for provided benchmarking event
#         r = requests.post(url=url, json=json_query, headers={'Content-Type': 'application/json'})
#         response = r.json()
#         data = response.get("data")
#         if data is None:
#             logging.fatal("For {} got response error from graphql query: {}".format(bench_event_id, r.text))
#             sys.exit(6)
#         if len(data["getChallenges"]) == 0:
#             logging.fatal("No challenges associated to benchmarking event " + bench_event_id +
#                           " in OEB. Please contact OpenEBench support for information about how to open a new challenge")
#             sys.exit()
#         else:
#             return data.get('getChallenges')
#     except Exception as e:

#         logging.exception(e)
        
# # function to populate bench_dir with existing aggregations
# def getOEBAggregations(response, output_dir):
#     for challenge in response:
        
#         challenge['datasets'][0]['datalink']["inline_data"] = json.loads(challenge['datasets'][0]["datalink"]["inline_data"])
        
#         for metrics in challenge['metrics_categories'][0]['metrics']:
#             if metrics['metrics_id'] == challenge['datasets'][0]['datalink']["inline_data"]["visualization"]["x_axis"]:
#                 challenge['datasets'][0]['datalink']["inline_data"]["visualization"]["x_axis"] = metrics['orig_id'].split(":")[-1]
#             elif metrics['metrics_id'] == challenge['datasets'][0]['datalink']["inline_data"]["visualization"]["y_axis"]:
#                 challenge['datasets'][0]['datalink']["inline_data"]["visualization"]["y_axis"] = metrics['orig_id'].split(":")[-1]
        
#         #replace tool_id for participant_id (for the visualitzation)
#         for i in challenge['datasets'][0]['datalink']['inline_data']['challenge_participants']:
#             i["participant_id"] = i.pop("tool_id")
        
#         new_aggregation = {
#             "_id": challenge['datasets'][0]['_id'],
#             "challenge_ids": [
#                  challenge['acronym']
#             ],
#             'datalink': challenge['datasets'][0]['datalink']
#         }
#         with open(os.path.join(output_dir, challenge['acronym']+".json"), mode='w', encoding="utf-8") as f:
#             json.dump(new_aggregation, f, sort_keys=True, indent=4, separators=(',', ': '))
  

def read_participant_data(participant_path):
    participant_data = {}

    with open(participant_path, mode='r', encoding="utf-8") as f:
        result = json.load(f)
        for item in result:
            participant_data.setdefault(item['challenge_id'], []).append (item)

    return participant_data

def find_all_challenge_benchmark_files(challenge,data_dir):
    challenges=[]
    for file in os.listdir(data_dir):
        print('this is file name: ',file)
        if file.startswith(challenge+'_') and file.endswith(".json"):
            challenges.append(file)
    
    print('these are challenges: ',challenges)
    return challenges

        
def generate_manifest(data_dir,output_dir, participant_data):

    info = []

    for challenge, metrics_file in participant_data.items():
        challenge_benchmark_files=find_all_challenge_benchmark_files(challenge,data_dir)
                
        challenge_dir = os.path.join(output_dir,challenge) # consolidation_out\disprot-disorder-pdb-atleast
        if not os.path.exists(challenge_dir):
            os.makedirs(challenge_dir)
        participants = set()
        
        challenge_oeb_data_dir = os.path.join(data_dir, challenge)  # benchmark_data\disprot-disorder-pdb-atleast

        
        for challenge_oeb_data in challenge_benchmark_files:
           
            if os.path.isfile(os.path.join(data_dir, challenge_oeb_data)):
                
                with open(os.path.join(data_dir, challenge_oeb_data)) as f:
                    aggregation_file = json.loads(f.read())
                    

                if aggregation_file["datalink"]["inline_data"]["visualization"]["type"]=="2D-plot":                    
                    metric_X = aggregation_file["datalink"]["inline_data"]["visualization"]["x_axis"] 
                    metric_Y = aggregation_file["datalink"]["inline_data"]["visualization"]["y_axis"]  
                else:
                    metric_X=aggregation_file["datalink"]["inline_data"]["visualization"]["x_axis"]

            # add new participant data to aggregation file
            new_participant_data = {}
            participant_id = '(unknown)'
            for metrics_data in metrics_file:
                print(f'the metrics_data is {metrics_data}')
                participant_id = metrics_data["participant_id"]

                if aggregation_file["datalink"]["inline_data"]["visualization"]["type"]=="2D-plot":
                    # print('metric_X is ',metric_X)
                    if metrics_data["metrics"]["metric_id"] == metric_X:
                        new_participant_data["metric_x"] = metrics_data["metrics"]["value"]
                        print('new_participant_data["metric_x"] ',new_participant_data["metric_x"] )
                    
                    elif metrics_data["metrics"]["metric_id"] == metric_Y:
                        print('metric_Y is ',metric_Y)

                        new_participant_data["metric_y"] = metrics_data["metrics"]["value"]
                        print('new_participant_data["metric_y"] ',new_participant_data["metric_y"])
                else:
                    if metrics_data["metrics"]["metric_id"] == metric_X:
                        new_participant_data["metric_x"] = metrics_data["metrics"]["value"]
                        print('new_participant_data["metric_x"] ',new_participant_data["metric_x"] )
            
            new_participant_data["participant_id"] = participant_id
            aggregation_file["datalink"]["inline_data"]["challenge_participants"].append(new_participant_data)
            participants.add(participant_id)           
            

            # add the rest of participants to manifest
            for name in aggregation_file["datalink"]["inline_data"]["challenge_participants"]:
                participants.add(name["participant_id"])

            #copy the updated aggregation file to output directory
            # print(f'challenge oeb data basename is {os.path.basename(challenge_oeb_data)}')
            summary_dir = os.path.join(challenge_dir,'summary_'+os.path.basename(challenge_oeb_data))
            # print(f'summary dir is : {summary_dir}')
            # print('challenge_dir is ',challenge_dir)
            # print('challenge is ',challenge)
            with open(summary_dir, 'w') as f:
                json.dump(aggregation_file, f, sort_keys=True, indent=4, separators=(',', ': '))
            

            if aggregation_file["datalink"]["inline_data"]["visualization"]["type"]=="2D-plot":
                assessment_chart.print_chart(challenge_dir, summary_dir, challenge, "RAW")
                assessment_chart.print_chart(challenge_dir, summary_dir, challenge, "SQR")
                assessment_chart.print_chart(challenge_dir, summary_dir, challenge, "DIAG")

            if aggregation_file["datalink"]["inline_data"]["visualization"]["type"]=="bar-plot":
                assessment_chart.print_barplot(challenge_dir, aggregation_file, challenge)

        #generate manifest
        obj = {
            "id" : challenge,
            "participants": list(participants),
            'timestamp': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        }

        info.append(obj)

    with open(os.path.join(output_dir, "Manifest.json"), mode='w', encoding="utf-8") as f:
        json.dump(info, f, sort_keys=True, indent=4, separators=(',', ': '))

        
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p", "--participant_data", help="path where the metrics data for the participant is stored", required=True)
    parser.add_argument("-b", "--benchmark_data", help="dir where the data for the benchmark will be or is stored", required=True)
    parser.add_argument("-o", "--output", help="output directory where the manifest and output JSON files will be written", required=True)
    # parser.add_argument("--offline", help="offline mode; existing benchmarking datasets will be read from the benchmark_data", required=False, type= bool)
    args = parser.parse_args()

    main(args)