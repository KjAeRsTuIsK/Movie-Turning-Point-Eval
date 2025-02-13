import json
import os
import argparse

# Define turning points and corresponding question ranges
TURNING_POINTS = ["Opportunity", "Change of Plans", "Point of No Return", "Major Setback", "Climax"]
QUESTION_RANGES = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]

def calculate_score(answers):
    """ Calculate the score based on the number of 'Yes' answers. """
    yes_count = answers.count("Yes")
    return {4: 10, 3: 7.5, 2: 5, 1: 2.5}.get(yes_count, 0)

def process_scene(scene):
    """ Process a single scene to calculate scores for turning points. """
    merged_answers = []
    
    for answer_set in scene["answer"]:
        answers = [line.split(":")[1] for line in answer_set.split("\n")]
        merged_answers.extend(answers)

    # Compute scores for each turning point
    return {
        "scene_number": scene["scene_number"],
        "answer": {
            TURNING_POINTS[i]: calculate_score(merged_answers[start:end])
            for i, (start, end) in enumerate(QUESTION_RANGES) if end <= len(merged_answers)
        }
    }

def process_folder(folder_path):
    """ Process all JSON files in a folder and save them in a new folder with '_changed_format' appended. """
    output_folder = f"{folder_path}_changed_format"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith(".json"):
            continue  # Skip non-JSON files
        
        with open(file_path, 'r') as f:
            input_data = json.load(f)

        processed_data = [{"scene_number": item["scene_number"], "answer": str(process_scene(item)["answer"])} for item in input_data]
        output_file_path = os.path.join(output_folder, file_name)
        with open(output_file_path, 'w') as out_file:
            json.dump(processed_data, out_file, indent=4)

        print(f"Processed: {file_name} -> Saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON files in a folder and save modified versions.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing JSON files.")
    args = parser.parse_args()

    process_folder(args.folder_path)
    print(f"Processing complete. Output saved in '{args.folder_path}_changed_format'.")
