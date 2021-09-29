import os
import json
import argparse

def merge_jsons(json_filenames_list,
                output_json_name="all_annotations.json"):
    """
    merges json files with names specified in filenames
    
    :param: filenames list containing file names
    :param: output_file, name of the final json
    """
    result = dict()
    for f in json_filenames_list:
        with open(f, 'r') as infile:
            result.update(json.load(infile))

    with open(output_json_name, 'w') as output_file:
        json.dump(result, output_file, indent=4)

    return

def merge_annotations(args):
    """
    Merges all jsons in a given folder to a single json, called "all_annotations". 
    folder_with_jsons: relative path to folder with jsons to be merged.
    """
    jsons = [f for f in os.listdir(args.path) if (f.endswith('.json') and not f.endswith("all_annotations.json"))]
    jsons = [os.path.join(args.path, j) for j in jsons]  # to relative paths 
    output_path = os.path.join(args.path, "all_annotations.json") # relative path 
    merge_jsons(jsons, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    subparsers = parser.add_subparsers()
    
    # create the parser for the "merge_annotations" command
    merge_annotations_parser = subparsers.add_parser("merge_annotations", help="merge annotations")
    merge_annotations_parser.set_defaults(func=merge_annotations)
    merge_annotations_parser.add_argument(
        "--path", type=str, required=True, help="folder path with jsons to be merged"
    )
    
    parser.set_defaults(func=merge_annotations)
    args = parser.parse_args()
    args.func(args)
