import argparse

from compute_metrics import write_metrics_file
from result_pred_ground_truth import results_to_pred_ground_truth_csv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=str, help="Author or method name")
    parser.add_argument("-f", type=str, help="Filename")
    args = parser.parse_args()
    author = args.a
    filename = args.f

    # Make a 4-column file with pred and ground truth, from participant results

    #file_pred_participant = f"{author}/submission_{filename}"
    file_pred_participant = f"{author}/{filename}"
    path_pred_ground_truth = f"{author}/results_pred_ground_truth_{filename}.csv"
    print(f"Generate results file from {file_pred_participant}")
    results_to_pred_ground_truth_csv(file_pred_participant, path_pred_ground_truth)
    print(f"File {path_pred_ground_truth} generated")

    # Write a file with metrics computed from results file

    output_path_metrics = f"{author}/metrics_{filename}.txt"
    print(f"Generate metrics file from {path_pred_ground_truth}")
    write_metrics_file(path_pred_ground_truth, output_path_metrics)
    print(f"File {output_path_metrics} generated")
