import argparse
import csv
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

def load_json(file_path):
    """ Load JSON data from a file. """
    with open(file_path, 'r') as file:
        return json.load(file)

def load_csv(file_path):
    """ Load CSV data into a list. """
    data_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            data_list.append(row)
    return data_list

def load_model(model_name):
    """ Load the pre-trained model and tokenizer. """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_answers(model, tokenizer, movie_name, summary, scene_summaries, shot_5_qa, few_shot_example_qa):
    """ Generate answers for each scene in the movie using a sliding window approach. """
    prompt_sliding_window_qa = (
        "There are six stages (acts) in a film, namely the setup, the new situation, progress, "
        "complications and higher stakes, the final push, and the aftermath, separated by five turning points (TPs). "
        "TPs are narrative moments from which the plot goes in a different direction, that is from one act to another. "
        "The five turning points are described as:\n"
        "1. Opportunity: An introductory event that occurs after presenting the setting and background of the main characters.\n"
        "2. Change of Plans: An event where the main goal of the story is defined, leading to increased action.\n"
        "3. Point of No Return: An event that pushes the main character(s) to fully commit to their goal.\n"
        "4. Major Setback: An event where everything falls apart (temporarily or permanently).\n"
        "5. Climax: The final event of the main story, resolving the main plot.\n\n"
        "You will be provided with the following inputs:\n"
        "- **Movie Summary**: A brief summary of the entire movie.\n"
        "- **Current Scene Summary**: The scene currently being analyzed.\n"
        "- **Scene Before Summary**: The scenes immediately preceding the current scene.\n"
        "- **Scene After Summary**: The scenes immediately following the current scene.\n"
        "Based on the provided context, answer the questions with Yes/No only. Do not provide reasoning."
    )

    generation_args = {
        "max_new_tokens": 256,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    final_movie_answers = []
    for i in tqdm(range(1, len(scene_summaries) - 1)):
        # start_index = max(0, i - 1)
        # end_index = min(len(scene_summaries), i + 2)
        # scenes_before = "\n".join(scene['summary'] for scene in scene_summaries[start_index:i])
        # scenes_after = "\n".join(scene['summary'] for scene in scene_summaries[i+1:end_index])

        messages_few_shot_prompt = [{"role": "system", "content": "You are an expert in movie understanding."}]

        # Include a few-shot example
        for index, item in enumerate(shot_5_qa):
            if index == 4:
                messages_few_shot_prompt.append({
                    "role": "user",
                    "content": prompt_sliding_window_qa + 
                               " MOVIE SUMMARY: " + item['movie_summary'] + 
                               " SCENE BEFORE: " + item['scene_before'] +
                               " PRESENT SCENE: " + item['scene_summary'] + 
                               " SCENE AFTER: " + item['scene_after'] + 
                               " Questions: \n" + 
                               item['o_q'] + '\n' + item['cop_q'] + '\n' +
                               item['ponr_q'] + '\n' + item['ms_q'] + '\n' + item['c_q']
                })
                messages_few_shot_prompt.append({
                    "role": "assistant",
                    "content": item['o_a'] + '\n' + item['cop_a'] + '\n' + item['ponr_a'] + '\n' + item['ms_a'] + '\n' + item['c_a']
                })

        messages_few_shot_prompt.append({
            "role": "user",
            "content": " MOVIE SUMMARY: " + summary + 
                       " SCENE BEFORE: " + scene_summaries[i-1]['summary'] + 
                       " PRESENT SCENE: " + scene_summaries[i]['summary'] + 
                       " SCENE AFTER: " + scene_summaries[i+1]['summary'] + 
                       " Questions: \n" + few_shot_example_qa[0]['o_q'] + '\n' + 
                       few_shot_example_qa[0]['cop_q'] + '\n' + few_shot_example_qa[0]['ponr_q'] + '\n' +
                       few_shot_example_qa[0]['ms_q'] + '\n' + few_shot_example_qa[0]['c_q']
        })

        output_answers = []
        output_0 = text_generator(messages_few_shot_prompt, **generation_args)
        output_answers.append(output_0[0]['generated_text'])

        final_movie_answers.append({
            "scene_number": scene_summaries[i]['scene_number'],
            "answer": output_answers
        })

    return final_movie_answers

def main(args):
    """ Main function to load data, process movies, and generate results. """
    # Login to Hugging Face
    login(token=args.hf_token)

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)

    # Load data
    movie_summaries = load_json(args.movie_summaries)
    shot_5_qa = load_json(args.few_shot_examples)
    all_movie_scenes_summary = load_json(args.scene_summaries)
    screenplay_data = load_csv(args.screenplays_file)

    # Process each movie
    for movie in tqdm(screenplay_data):
        movie_name = movie[0]

        summary = next((m['summary'] for m in movie_summaries if m['movie_name'] == movie_name), None)
        scene_summaries = next((m['all_summaries'] for m in all_movie_scenes_summary if m['movie_name'] == movie_name), None)

        if not summary or not scene_summaries:
            continue

        final_movie_answers = generate_answers(model, tokenizer, movie_name, summary, scene_summaries, shot_5_qa, shot_5_qa)

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f'answers_{movie_name}.json')

        with open(output_file, 'w') as f:
            json.dump(final_movie_answers, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process movie screenplays to identify turning points.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument('--movie_summaries', type=str, required=True, help="Path to the movie summaries JSON file.")
    parser.add_argument('--few_shot_examples', type=str, required=True, help="Path to the 5-shot QA JSON file.")
    parser.add_argument('--scene_summaries', type=str, required=True, help="Path to the movie scenes summaries JSON file.")
    parser.add_argument('--screenplays_file', type=str, required=True, help="Path to the screenplays CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output JSON files.")
    args = parser.parse_args()

    main(args)
