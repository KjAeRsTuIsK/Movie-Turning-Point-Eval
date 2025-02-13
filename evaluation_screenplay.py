
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy.stats import norm
import csv
import ast
import argparse

def read_jsonl_file(file_path):
    scenes = []
    scores = []
    with open(file_path, 'r') as file:
        data_input=json.load(file)
        for i, line in enumerate(data_input):
            
            # import pdb;pdb.set_trace()
            input_sentence=line['answer'].split("Reasoning:")[0]
            input_sentence=line['answer'].replace("'","\"")

            if ("\"Opportunity\"") not in input_sentence:
                
                input_sentence=line['answer'].strip().replace("Oppurtunity","\"Opportunity\"").replace("Change of Plans","\"Change of Plans\"").replace("Point of No Return","\"Point of No Return\"").replace("Major Setback","\"Major Setback\"").replace("Climax","\"Climax\"").replace("None","\"None\"")
                # print(input_sentence)
            # line=line.replace('"{','{').replace('}"','}').replace('\\','')
            # print(line)
            # exit()
            json_matches = re.findall(r'\{[^}]+\}', input_sentence)
            # print(json_matches)
            if json_matches:
                json_str = json_matches[0]
                json_str=json_str.replace('"{','{').replace('}"','}').replace('\\','')
                # print(json_str)
                # exit()
                try:
                    scene_data = json.loads(json_str)
                    scenes.append(line['scene_number'])  # Scene number (1-indexed)
                    scores.append([
                        scene_data.get('Opportunity', 0),
                        scene_data.get('Change of Plans', 0),
                        scene_data.get('Point of No Return', 0),
                        scene_data.get('Major Setback', 0),
                        scene_data.get('Climax', 0),
                        scene_data.get('None', 0),
                    
                    ])
                    # print(scene_data)
                    # print(scores)
                    # import pdb;pdb.set_trace()
                except json.JSONDecodeError:
                    print(f"Warning: Unable to parse JSON in line {i+1}. Skipping this line.")
            else:
                print(f"Warning: No valid JSON object found in line {i+1}. Skipping this line.")
    return scenes, scores

def apply_range_constraints(matrix, mean_percentages, std_percentages):
    # Initialize the new matrix with zeros
    new_matrix = np.zeros_like(matrix, dtype=float)
    
    total_rows = matrix.shape[0]
    
    # Convert percentages to absolute values
    means = np.array([mean * total_rows / 100 for mean in mean_percentages])
    std_devs = np.array([std * total_rows / 100 for std in std_percentages])
    
    # Iterate over each column (turning point)
    for col, (mean, std) in enumerate(zip(means, std_devs)):
        # Define the range based on percentage range (assumed as ±1 standard deviation)
        lower_bound = int(max(0, mean - std))
        upper_bound = int(min(total_rows, mean + std))
        
        # Ensure bounds are within the valid range
        lower_bound = max(0, lower_bound)
        upper_bound = min(total_rows, upper_bound)
        # print(lower_bound,upper_bound)
        # Copy the values within the range to the new matrix
        if lower_bound < upper_bound:
            new_matrix[lower_bound:upper_bound, col] = matrix[lower_bound:upper_bound, col]
    
    return new_matrix

def apply_normal_distribution(matrix, column_stats):
    # Initialize the new matrix with zeros
    new_matrix = np.zeros_like(matrix, dtype=float)
    
    # Define the min and max range
    min_val = 0
    max_val = 10
    
    total_rows = matrix.shape[0]
    
    # Iterate over each column
    for col, (mean, std) in enumerate(column_stats):
        # Generate a normal distribution over the range [0, 10]
        x = np.linspace(min_val, max_val, total_rows)
        probabilities = norm.pdf(x, mean, std)
        # print(probabilities)
        # Normalize the probabilities so that they sum to 1
        # probabilities /= probabilities.sum()
        
        # Multiply the column values by the normalized probabilities
        new_matrix[:, col] = matrix[:, col] * probabilities
    new_matrix = new_matrix.astype(int)
    return new_matrix

def calculate_predicted_turning_points(scenes, scores, num_turning_points,ideal_positions):
    num_scenes = len(scenes)

    # Initialize the DP table
    dp = [[float('inf')] * (num_turning_points + 1) for _ in range(num_scenes + 1)]
    dp[0][0] = 0
    
    # Fill the DP table
    for i in range(1, num_scenes + 1):
        dp[i][0] = 0
        for j in range(1, min(i, num_turning_points) + 1):
            # Option 1: Don't use this scene as a turning point
            dp[i][j] = dp[i-1][j]
            
            # Option 2: Use this scene as the j-th turning point
            score = scores[i-1][j-1]
            cost = (10 - score)   # Higher confidence means lower cost
            dp[i][j] = min(dp[i][j], dp[i-1][j-1] + cost)
    
    # Backtrack to find the optimal turning points
    
    optimal_turning_points = [[] for _ in range(num_turning_points)]
    i, j = num_scenes, num_turning_points
    path=[]
    while j > 0:
        if i > 0 and dp[i][j] != dp[i-1][j]:
            optimal_turning_points[j-1].append(scenes[i-1])
            path.append((i,j,'D'))
            i -= 1
            j -= 1
        else:
            path.append((i,j,'u'))
            i -= 1
    print(optimal_turning_points)
 
    dp_array = np.array(dp)

    # Handle cases where a turning point does not exist
    for i in range(num_turning_points):
        if not optimal_turning_points[i]:
            print(f"Movie Name: <MovieNameHere>, Turning Point: {i}, Confidence Score: 0")
    
    return [list(reversed(points)) for points in optimal_turning_points], dp_array


def expand_score_matrix(scores):
    # Repeat each column of the original score matrix 3 times
    expanded_scores = np.repeat(scores, 3, axis=1)
    return expanded_scores

def calculate_turning_points_max_score_based(scores, scenes,threshold):
    """
    Calculate the turning points based on confidence scores without any constraints.

    Parameters:
    scores (list of lists): Confidence scores for each scene and turning point.
    scenes (list): List of scene indices.

    Returns:
    dict: Dictionary with turning point names as keys and corresponding scene indices as values.
    """
    num_turning_points = 5  # Assuming each scene has scores for all turning points
    turning_points = [[] for _ in range(num_turning_points)]

    for scene_idx, scene in enumerate(scenes):
        for tp_idx in range(num_turning_points):
            if scores[scene_idx][tp_idx] >= threshold:
                turning_points[tp_idx].append(scene)

    return turning_points
   

def get_ground_truth_for_movie(file_path, movie_name):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row)
            if row['movie_name'] == movie_name:
                tp1 = set(ast.literal_eval(row['tp1']))
                tp2 = set(ast.literal_eval(row['tp2']))
                tp3 = set(ast.literal_eval(row['tp3']))
                tp4 = set(ast.literal_eval(row['tp4']))
                tp5 = set(ast.literal_eval(row['tp5']))
                
                return [tp1, tp2, tp3, tp4, tp5]
    
    return None  # If the movie name is not found in the CSV file

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
        pred_set_expanded=expand_set(pred_set)
        # print(pred_set,pred_set_expanded)
        # intersection=pred_set & gt_set
        ##According to the original paper, we expand the prediction set to allow for +-1 scenes to be considered correct
        intersection = pred_set_expanded & gt_set
        union = pred_set | gt_set
        total_agreement+=len(intersection)/len(union)
        # if len(intersection) == len(union):
        #     total_agreement += 1
        
        # Partial Agreement (PA): Check if there is any overlap between the sets
        if intersection:
            partial_agreement += 1
        
        # Annotation Distance (D): Mean of minimum distances between predicted and ground truth sets
        if pred_set and gt_set:
            min_distances = [min(abs(pred - gt) for gt in gt_set) for pred in pred_set_expanded]
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
        num_scenes=pred['num_scenes']
        predicted_tp = [set(tp) for tp in pred['turning_points']]
        ground_truth_tp = [set(tp) for tp in gt['turning_points']]
        new_predicted_tp=[expand_set(x) for x in predicted_tp]
        TA, PA, D = compute_agreement(predicted_tp, ground_truth_tp)
        D=D/num_scenes
        print(f"Movie: {movie_name}")
        print(f"Total Agreement (TA): {TA:.4f}")
        print(f"Partial Agreement (PA): {PA:.4f}")
        print(f"Annotation Distance (D): {D:.4f}")
        print()

        total_TA += TA
        total_PA += PA
        total_D += D

    print("Average Results Across All Movies:")
    print(f"Total Agreement (TA): {total_TA / num_movies:.4f}")
    print(f"Partial Agreement (PA): {total_PA / num_movies:.4f}")
    print(f"Annotation Distance (D): {total_D / num_movies:.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process screenplay data and evaluate turning points.")
    parser.add_argument("--gt_file", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the JSON file containing input data.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSON file for predicted TPs. ")
    return parser.parse_args()

def main():
    args=parse_arguments()
    folder_path=args.input_folder
    movie_files=os.listdir(folder_path)
    answer_file=args.gt_file

    
        
    final_predicted_tp=[]
    ground_truth_tp=[]
    
    for file in movie_files:  
          
        file_path=os.path.join(folder_path,file)
        movie_name=file.split('answers_')[1].replace('.json','')
        print(movie_name)
        ideal_positions=get_ground_truth_for_movie(answer_file,movie_name)
        # ideal_positions=get_ground_truth_for_movie_sentence(answer_file,movie_name)

        scenes, scores = read_jsonl_file(file_path)
        scores_mat=np.array(scores)
        
        
        #######     To remove range contraints, comment the next line
        scores_mat=apply_range_constraints(scores_mat,[11.39,31.86,50.65,74.15,89.43 ],[6.72, 11.26, 12.15, 8.40, 4.74])
        #######
        
        num_scenes=len(scores)+2 #Because we skip first scene and last scene while running the DP+QA method


        predicted_turning_points,dp_array = calculate_predicted_turning_points(scenes, scores_mat, 5,ideal_positions)
        
        ground_truth_tp.append({
                    "movie_name": movie_name,
            "turning_points": ideal_positions
        })
        final_predicted_tp.append({
            "movie_name": movie_name,
            "turning_points": predicted_turning_points,
            "num_scenes":num_scenes
        })

    print(final_predicted_tp)
    print(ground_truth_tp)
    evaluate_movies(final_predicted_tp,ground_truth_tp)
    with open(args.output_file,'w') as f:
        json.dump(final_predicted_tp)


if __name__ == "__main__":
    main()
