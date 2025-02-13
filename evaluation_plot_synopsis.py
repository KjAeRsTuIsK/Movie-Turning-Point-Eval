
import json
import argparse
import re
import os
from scipy.stats import norm
import pandas as pd
import numpy as np
import csv
import ast


def get_ground_truth_for_movie_sentence(file_path, movie_name):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['movie_name'] == movie_name:
                tp1 = {int(row['tp1'])} if row['tp1'] else set()
                tp2 = {int(row['tp2'])} if row['tp2'] else set()
                tp3 = {int(row['tp3'])} if row['tp3'] else set()
                tp4 = {int(row['tp4'])} if row['tp4'] else set()
                tp5 = {int(row['tp5'])} if row['tp5'] else set()
                
                return [tp1, tp2, tp3, tp4, tp5]

def read_ground_truth_csv(file_path):
    df = pd.read_csv(file_path)
    ground_truth = {}
    
    for _, row in df.iterrows():
        movie_name = row['movie_name']
        ground_truth[movie_name] = {
            'tp1': ast.literal_eval(row['tp1']),
            'tp2': ast.literal_eval(row['tp2']),
            'tp3': ast.literal_eval(row['tp3']),
            'tp4': ast.literal_eval(row['tp4']),
            'tp5': ast.literal_eval(row['tp5'])
        }
    
    return ground_truth


def expand_set(original_set):
    """
    Expand each element of the set to include ±1.

    Parameters:
    original_set (set): The set of predicted turning point indices.

    Returns:
    set: Expanded set including ±1 of each element.
    """
    expanded = set()
    for element in original_set:
        expanded.add(element)
        expanded.add(element - 1)
        expanded.add(element + 1)
    return expanded

def compute_agreement(predicted_tp, ground_truth_tp):
    """
    Compute the agreement between predicted turning points and ground truth.

    Parameters:
    predicted_tp (list of sets): List of sets of predicted turning point indices.
    ground_truth_tp (list of sets): List of sets of possible ground truth indices for each turning point.

    Returns:
    tuple: Total Agreement (TA), Partial Agreement (PA), Annotation Distance (D).
    """
    total_agreement = 0
    partial_agreement = 0
    annotation_distances = []

    for pred_set, gt_set in zip(predicted_tp, ground_truth_tp):
        # Total Agreement (TA): Intersection over Union for sets
        intersection=pred_set & gt_set
        union = pred_set | gt_set
        total_agreement+=len(intersection)/len(union)
        # if len(intersection) == len(union):
        #     total_agreement += 1
        
        # Partial Agreement (PA): Check if there is any overlap between the sets
        if intersection:
            partial_agreement += 1
        
        # Annotation Distance (D): Mean of minimum distances between predicted and ground truth sets
        if pred_set and gt_set:
            min_distances = [min(abs(pred - gt) for gt in gt_set) for pred in pred_set]
            annotation_distances.append(sum(min_distances) / len(min_distances))

    T = len(predicted_tp)  # Total number of turning points

    # Compute average values
    TA = total_agreement / T
    PA = partial_agreement / T
    D = sum(annotation_distances) / T if annotation_distances else 0

    return TA, PA, D


def evaluate_movies(predictions, ground_truth):
    """
    Evaluate multiple movies and print the results.

    Parameters:
    predictions (list): List of dictionaries containing predicted turning points.
    ground_truth (list): List of dictionaries containing ground truth turning points.
    """
    total_TA, total_PA, total_D = 0, 0, 0
    num_movies = len(predictions)

    for pred, gt in zip(predictions, ground_truth):
        movie_name = pred['movie_name']
        num_sentences=pred['num_sentences']
        predicted_tp = [set(tp) for tp in pred['turning_points']]
        ground_truth_tp = [set(tp) for tp in gt['turning_points']]
        new_predicted_tp=[expand_set(x) for x in predicted_tp]
        TA, PA, D = compute_agreement(predicted_tp, ground_truth_tp)
        D=D/num_sentences
        # print(f"Movie: {movie_name}")
        # print(f"Total Agreement (TA): {TA:.4f}")
        # print(f"Partial Agreement (PA): {PA:.4f}")
        # print(f"Annotation Distance (D): {D:.4f}")
        # print()

        total_TA += TA
        total_PA += PA
        total_D += D

    print("Average Results Across All Movies:")
    print("For Plot Synopsis, TA=PA !!!!!!!")
    print(f"Total Agreement (TA): {total_TA / num_movies:.4f}")
    print(f"Partial Agreement (PA): {total_PA / num_movies:.4f}")
    print(f"Annotation Distance (D): {total_D / num_movies:.4f}")

def extract_turning_points(text):
    # Use regex to find sentence numbers after "Turning Point X (Sentence Y):"
    return list(map(int, re.findall(r'Turning Point \d+ \(Sentence (\d+)\):', text)))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process screenplay data and evaluate turning points.")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing input data.")
    parser.add_argument("--separate_sentences", type=str, required=True, help="Path to the JSON file containing separate sentences.")
    return parser.parse_args()


def main():
    args=parse_arguments()
    gt_file=args.gt_file
    
    with open(args.input_file,'r') as f:
        input_data=json.load(f)
        
    with open(args.separate_sentences,'r') as f:
        input_data_sentences=json.load(f)
        
        
    final_predicted_tp=[]
    ground_truth_tp=[]
    for item in input_data:    
        
        movie_name=item['movie_name']
        ideal_positions=get_ground_truth_for_movie_sentence(gt_file,movie_name)
        predicted_turning_points_input=next((movie['summary'] for movie in input_data if movie['movie_name'] == movie_name), "Movie not found.")
        num_sentences=next((movie['total_sentences'] for movie in input_data_sentences if movie['movie_name'] == movie_name), "Movie not found.")
        predicted_turning_points_input=predicted_turning_points_input[0].split('<|start_header_id|>assistant<|end_header_id|>\n\n')[1].replace("<|eot_id|>",'')

        numbers=extract_turning_points(predicted_turning_points_input)
        predicted_turning_points = [[num] for num in numbers]

        ground_truth_tp.append({
                    "movie_name": movie_name,
            "turning_points": ideal_positions,
        })
        final_predicted_tp.append({
            "movie_name": movie_name,
            "turning_points": predicted_turning_points,
           "num_sentences":num_sentences

        })
    
    
    print(final_predicted_tp)
    print(ground_truth_tp)
    evaluate_movies(final_predicted_tp,ground_truth_tp)

if __name__ == "__main__":
    main()
