import argparse
import csv
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

# Set default tensor type to HalfTensor for GPU optimization
torch.set_default_tensor_type(torch.cuda.HalfTensor)

def load_model_and_tokenizer(model_name):
    """
    Load the pre-trained model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def load_csv(file_path):
    """
    Load data from a CSV file into a list.
    """
    data_list = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            data_list.append(row)
    return data_list

def generate_turning_points(model, tokenizer, scene_summaries):
    """
    Generate turning points for a given movie's scene summaries.
    """
    prompt = (
        "There are six stages (acts) in a film: the setup, the new situation, progress, "
        "complications and higher stakes, the final push, and the aftermath, separated by five turning points (TPs). "
        "The five turning points are:\n"
        "1. Opportunity: An introductory event that drives the narrative from setup to new situation.\n"
        "2. Change of Plans: An event defining the main goal, leading to increased action.\n"
        "3. Point of No Return: An event pushing the main character(s) to fully commit to their goal.\n"
        "4. Major Setback: An event where everything falls apart, leading to the final push.\n"
        "5. Climax: The final event of the main story, resolving the main plot.\n\n"
        "You will be provided with the plot synopsis of a movie. Your task is to find all 5 turning points. "
        "The output should be the sentence number for each turning point, starting from 0. "
        "###Example Output###: Turning points: {1,3,15,20,31}, where each number is the sentence number from the plot synopsis. "
        "Do not provide reasoning for the answer."
    )

    input_synopsis = ""
    for sentence in scene_summaries:
        input_synopsis += f"\nSentence {sentence['sentence_number']}:\n{sentence['sentence']}"

    messages = [
        {"role": "system", "content": "You are an expert at movie understanding."},
        {"role": "user", "content": prompt + "\nPLOT SYNOPSIS:\n" + input_synopsis},
    ]

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

    output = text_generator(messages, **generation_args)
    return output[0]['generated_text']

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process movie data to identify turning points.")
    parser.add_argument('separate_sentences_json', type=str, help="Path to the separate sentence JSON file.")
    parser.add_argument('screenplays_csv_path', type=str, help="Path to the screenplays CSV file.")
    # parser.add_argument('synopses_csv_path', type=str, help="Path to the synopses CSV file.")
    parser.add_argument('output_json_path', type=str, help="Path to save the output JSON file.")
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load data
    all_sentences = load_json(args.separate_sentences_json)
    screenplays_data = load_csv(args.screenplays_csv_path)
    # synopses_data = load_csv(args.synopses_csv_path)

    all_movie_summaries = []

    for screenplay in tqdm(screenplays_data):
        movie_name = screenplay[0]

        # Find scene summaries for the current movie
        scene_summaries = next(
            (movie['sentences'] for movie in all_sentences if movie['movie_name'] == movie_name),
            None
        )

        if scene_summaries:
            output_answer = generate_turning_points(model, tokenizer, scene_summaries)
            all_movie_summaries.append({
                "movie_name": movie_name,
                "answer": output_answer
            })

    # Save the results to a JSON file
    with open(args.output_json_path, 'w') as f:
        json.dump(all_movie_summaries, f, indent=4)

if __name__ == "__main__":
    main()

    

 
            
        # prompt_summary="Your task is to take the detailed movie screenplay provided below\
        #     and convert it into a concise summary. The summary should capture the main plot points and key events of the movie in 9-10 sentences. Output should be the summary only. Do not write anything else. Do not provide any headings or mention any scene numbers. Screenplay: "
        
        # prompt_scene_summary=" Your task is to summarize the given scene and convert it into a concise summary.The summary should capture the main plot points and key events of the scene. Just output the summary, without any reasoning or your own thoughts on the story."

        # prompt_dp_qa = (
        # "There are six stages (acts) in a film, namely the setup, the new situation, progress, complications and higher stakes, the final push, and the aftermath, separated by five turning points (TPs). TPs are narrative moments from which the plot goes in a different direction, that is from one act to another. The five turning points are described as:\n"
        # "1. Opportunity: An introductory event that occurs after presenting the setting and background of the main characters, driving the narrative from setup to new situation.\n"
        # "2. Change of Plans: An event where the main goal of the story is defined, leading to increased action, driving the narrative from the introduction of new situation to it's progress.\n"
        # "3. Point of No Return: An event that pushes the main character(s) to fully commit to their goal, driving the narrative from the progress to it's complications and high-stake difficulties.\n"
        # "4. Major Setback: An event where everything falls apart (temporarily or permanently), progressing the story from complications to the final push. \n"
        # "5. Climax: The final event of the main story, the moment of resolution, and the 'biggest spoiler', progressing the story from the final push to the aftermath of the main plot.\n\n"
        # # "For each present scene, provide the confidence score (from 1 to 10) that it belongs to each turning point.\n"
        # # "Format your answer as: \n"
        
        # "You will be provided with the following inputs: 1. **Movie Summary**: A brief summary of the entire movie, providing context for the overall plot and character arcs."
        # "2. **Current Scene Summary**: A summary of the scene currently being analyzed."
        # "3. **Scene Before Summary**: A summary of the scenes immediately preceding the current scene."
        # "4. **Scene After Summary**: A summary of the scenes immediately following the current scene."
        # "Based on the movie summary, the summaries for the scenes before and after the present scenes, you will be asked questions related to identifying turning points in the plot synopsis. Answer the questions with a Yes/No."
        # "Do not provide reasoning for the answer. ")
        
        # few_shot_example_qa=[{'gt_answer': 'Major Setback',
        # 'tp4': '87',
        # 'movie_name': 'Juno (film)',
        # 'movie_summary': "Juno, a 16-year-old high school student, discovers she is pregnant after a one-night stand with her best friend Bleeker. She decides to give the baby up for adoption and starts looking for a suitable family. She meets Mark and Vanessa Loring, a couple who seem perfect, but Juno's feelings for Bleeker grow stronger as she gets to know them. Meanwhile, Bleeker is struggling with his own feelings for Juno and the impending arrival of the baby. As Juno's due date approaches, she must make a difficult decision about the baby's future and her own relationships. The film explores themes of teenage pregnancy, adoption, and first love, with a quirky and offbeat tone. Juno's journey is marked by her wit, sarcasm, and determination, as she navigates the complexities of growing up and finding her place in the world. Ultimately, the film ends on a hopeful note, with Juno and Bleeker reuniting and starting a new chapter in their lives.",
        # 'movie_scene': '====================87====================\n               INT. LORING HOUSE - BASEMENT - AFTERNOON 87\n\n\n\n               The Lorings\' basement is dank, cluttered unfinished and \n\n               unattractive, much like Paulie Bleeker\'s. Mark pulls a chain \n\n               to illuminate a bare bulb.\n\n\n\n                                     JUNO\n\n                         Oh, Mark! Is this the baby\'s room? \n\n                         It\'s beautiful!\n\n\n\n                                     MARK\n\n                         Hilarious. No, I just keep all of my \n\n                         old comics down here, and I want to \n\n                         show you one of them.\n\n\n\n                                     JUNO\n\n                         Oh God, you\'re one of those guys...\n\n\n\n                                     MARK\n\n                         You\'re gonna like this, I promise.\n\n\n\n               Mark RUMMAGES through a cardboard box in the corner.\n\n\n\n                                     MARK\n\n                              (extracting a bagged \n\n                              COMIC from the box)\n\n                         Here it is.\n\n\n\n               He shows the COMIC to Juno. It\'s called "Most Fruitful Yuki." \n\n               It depicts a pregnant JAPANESE GIRL kicking ass and taking \n\n               names.\n\n\n\n                                     JUNO\n\n                         "Most Fruitful Yuki"? What is... Oh \n\n                         my god, she\'s a pregnant superhero!\n\n\n\n                                     MARK\n\n                         Isn\'t that great? I got it when I \n\n                         was in Japan with my band. She reminds \n\n                         me of you.\n\n\n\n               Juno examines the comic. "Most Fruitful Yuki" does resemble \n\n               her.\n\n\n\n                                     JUNO\n\n                         Wow, I actually feel like less of a \n\n                         fat dork now.\n\n\n\n                                     MARK\n\n                         Most Fruitful Yuki is bad ass, man. \n\n                         You should be proud to be the same \n\n                         condition.\n\n\n\n               She throws a KARATE KICK in Mark\'s direction with a KEE-YA!\n\n\n\n               Juno is sincerely pleased.\n\n\n\n                                     JUNO\n\n                         Okay, how about some tunes?\n\n\n\n               There\'s a battered portable CD player in the corner. Juno \n\n               kneels down and pops in one of the discs.\n\n\n\n                                     JUNO\n\n                         Now this first one is kinda slow. \n\n                         But it\'s Mott the Hoople so it\'s \n\n                         still totally rad and hardcore.\n\n\n\n               Juno puts in the CD and "All The Young Dudes" fills the room.\n\n\n\n               Mark Laughs.\n\n\n\n                                     JUNO\n\n                         What?\n\n\n\n                                     MARK\n\n                         I actually know this one.\n\n\n\n                                     JUNO\n\n                         You do?\n\n\n\n                                     MARK\n\n                         Yeah, this song\'s older than me, if \n\n                         you can believe that. I danced to it \n\n                         at my senior prom.\n\n\n\n                                     JUNO\n\n                         That\'s almost interesting, Mark. Who \n\n                         did you dance with?\n\n\n\n                                     MARK\n\n                         Her name was Cynthia Vogel and she \n\n                         was a good dance partner. Even let \n\n                         me put my hands on her butt.\n\n\n\n                                     JUNO\n\n                         Oh man, I can just picture you slow \n\n                         dancing like a dork!\n\n\n\n               She mockingly places her hands on Mark\'s waist and moves \n\n               back and forth stiffly.\n\n\n\n                                     MARK\n\n                         No, I put my hands on your waist. \n\n                         Then you put your arms around my \n\n                         neck. That\'s how we did it in \'88.\n\n\n\n               Mark puts his hands on what remains of Juno\'s waist. She \n\n               drapes her arms around his neck self-consciously.\n\n\n\n                                     JUNO\n\n                         Oh, okay. Like this.\n\n\n\n                                     MARK\n\n                         You\'ve never been to a dance, have \n\n                         you?\n\n\n\n                                     JUNO\n\n                              (casually defensive)\n\n                         Only squares and nerds go to dances.\n\n\n\n                                     MARK\n\n                         What are you?\n\n\n\n                                     JUNO\n\n                         I don\'t know.\n\n\n\n               They SWAY slowly to the music. Juno\'s belly bumps up against \n\n               Mark.\n\n\n\n                                     MARK\n\n                         I feel like there\'s something between \n\n                         us.\n\n\n\n               They laugh.\n\n\n\n               Juno rests her head on Mark\'s chest. They dance in silence \n\n               for a few moments, then stop moving. Mark pulls Juno as close \n\n               as he possibly can, given her expanding girth.\n\n\n\n                                     MARK\n\n                         I\'m leaving Vanessa.\n\n\n\n                                     JUNO\n\n                              (quiet at first)\n\n                         What?\n\n\n\n                                     MARK\n\n                         It\'s just not working out, but I\'m \n\n                         getting my own place in the city... \n\n                         and I\'ve got it all planned out. \n\n                         It\'s something I\'ve wanted to do for \n\n                         a long time...\n\n\n\n               Juno backs away.\n\n\n\n                                     JUNO\n\n                              (growing)\n\n                         No.\n\n\n\n                                     MARK\n\n                         No?\n\n\n\n                                     JUNO\n\n                         No. No, you definitely cannot do \n\n                         that, Mark. That\'s a big, fat sack \n\n                         of no!\n\n\n\n                                     MARK\n\n                         What\'s the matter?\n\n\n\n                                     JUNO\n\n                         This isn\'t what we agreed on. You \n\n                         guys have to take care of... this! \n\n                         You are the chosen custodians of the \n\n                         big-ass bump!\n\n\n\n               She GESTURES wildly to her belly. Suddenly, something matters \n\n               to her far more than the approval of an older guy.\n\n\n\n                                     MARK\n\n                         But I thought you\'d be cool if...\n\n\n\n                                     JUNO\n\n                              (interrupting)\n\n                         I want you guys to adopt the Buglet. \n\n                         I wanted everything to be perfect. \n\n                         Not shitty and broken like everyone \n\n                         else\'s family. Listen, once I have \n\n                         the baby, Vanessa is going to finally \n\n                         be happy, and everything will be all \n\n                         right. Believe me on this one!\n\n\n\n                                     MARK\n\n                         A baby is not going to fix everything. \n\n                         Besides, I don\'t know if I\'m ready \n\n                         to be a father.\n\n\n\n                                     JUNO\n\n                              (aghast)\n\n                         But you\'re old!\n\n\n\n                                     MARK\n\n                         I... How do you think of me, Juno? \n\n                         Why are you here?\n\n\n\n                                     JUNO\n\n                         I don\'t know. I just liked being \n\n                         your friend. I sort of liked becoming \n\n                         furniture in your weird life.\n\n\n\n                                     MARK\n\n                         This...\n\n                              (he gestures to the \n\n                              dank surrounding \n\n                              room)\n\n                         ...this is what my life has become. \n\n                         Stuff in boxes. Stuff underground. \n\n                         Is that so appealing to you?\n\n\n\n                                     JUNO\n\n                         Yeah, I guess... Is this my fault? \n\n                         Is Vanessa mad at you because of me?\n\n\n\n                                     MARK\n\n                         That\'s not the point. We\'re just not \n\n                         in love anymore.\n\n\n\n                                     JUNO\n\n                         Yeah, but didn\'t you love Vanessa \n\n                         when you married her? If you love \n\n                         someone once, you can love them again, \n\n                         I know it. My friend Leah has gone \n\n                         out with the same guy, like, four \n\n                         times. You\'re just not trying.\n\n\n\n               Mark suddenly sees Juno for what she is -- a teenage girl.\n\n\n\n                                     MARK\n\n                         I\'m such an idiot. I can\'t believe \n\n                         what an idiot I am.\n\n\n\n               He paces over to the wall and KICKS it softly.\n\n\n\n                                     JUNO\n\n                         Please don\'t get a divorce! God, \n\n                         Mark, just do me a solid and stay \n\n                         with your wife.\n\n\n\n                                     MARK\n\n                         God, you\'re so young.\n\n\n\n                                     JUNO\n\n                         Not really. I\'m sixteen. I\'m old \n\n                         enough to tell when people are acting \n\n                         like total a-holes!\n\n\n\n               Juno turns to leave, then shoots one furious look back at \n\n               Mark.\n\n\n\n                                     JUNO\n\n                         Oh and by the way, I bought another \n\n                         Sonic Youth album and it\'s the worst \n\n                         thing I\'ve ever heard! It\'s just \n\n                         noise, man!\n\n\n\n               She bolts up the stairs, sobbing.',
        # 'scene_summary': 'Juno visits Mark\'s basement, where he shows her a comic book called "Most Fruitful Yuki," a pregnant Japanese superhero. Juno is pleased to see a character like herself in the comic. They bond over music, dancing to Mott the Hoople\'s "All the Young Dudes." Mark reveals he\'s leaving his wife, Vanessa, to start a new life in the city, but Juno is opposed to the idea, insisting that Mark and Vanessa should adopt the baby she\'s carrying. Mark is hesitant, and Juno accuses him of being an "idiot" for not trying to work things out with his wife. Juno leaves the basement, upset and sobbing, after a heated argument with Mark.',
        # 'scene_before': "Mark opens the door to find Juno, visibly pregnant, standing outside. He greets her with a smile, complimenting her shirt. Juno inquires about Vanessa's presence, and Mark assures her they are alone. He invites Juno into the house, suggesting he has something for her.",
        # 'scene_after': "Juno arrives at the Loring house, but is intercepted by Vanessa, who senses something is wrong. Juno denies crying, but Vanessa's concern is evident. Mark tries to downplay the situation, claiming Juno is hormonal, but Vanessa knows he's lying. \n\nMark reveals he's having doubts about becoming a father, feeling unprepared and uncertain about the timing. Vanessa tries to reassure Juno that Mark will come around, but Juno is unconvincing. The conversation escalates, with Mark expressing his fears and Vanessa becoming increasingly frustrated. \n\nVanessa confronts Mark about his lack of commitment, telling him to grow up and stop being indecisive. Mark admits he's not going to be a great father, and the conversation ends with Juno escaping the conflict by leaving the house.",
        # 'o_q': ' Q1: Does this scene introduce a new element or character that changes the direction of the story?,\n        Q2: Is this scene significant in setting up the main conflict or goal of the narrative?,\n        Q3: Does this scene occur after the initial setup and begin driving the story towards the new situation?,\n        Q4: Is this scene crucial in moving the plot from the exposition phase to a new narrative phase?',
        # 'o_a': 'A1:Yes\nA2:Yes\nA3:No\nA4:No',
        # 'cop_q': "Q5: Does this scene clarify or redefine the main character's goal or motivation?,\n        Q6: Is there a change in the character's plan or approach to solving their main problem in this scene?,\n        Q7: Is there an unexpected twist that forces adaptation?,\n        Q8: Is this scene pivotal in escalating the narrative towards higher stakes or new challenges?",
        # 'cop_a': '\nA5:No\nA6:Yes\nA7:Yes\nA8:No',
        # 'ponr_q': "Q1: Does this scene involve a major commitment or irreversible decision by the character?,\n        Q2: Is there an evident shift in the character's behavior or attitude as a result of this scene?,\n        Q3: Does the scene show the character fully committing to a course of action despite risks or consequences?,\n        Q4: Is there a clear transition where the character's previous options or paths are no longer viable?",
        # 'ponr_a': 'A1:No\nA2:No\nA3:No\nA4:No',
        # 'ms_q': "Q5: Does this scene represent a significant failure or obstacle for the main character?,\n        Q6: Is there a dramatic downturn or crisis that affects the character's goals or plans?,\n        Q7: Does this scene illustrate a temporary or permanent collapse of the character's progress?,\n        Q8: Is this scene a key moment where the narrative takes a turn towards increased difficulty or tension?",
        # 'ms_a': '\nA5:Yes\nA6:Yes\nA7:Yes\nA8:Yes',
        # 'c_q': 'Q9: Does this scene represent the highest point of tension or conflict in the story?,\n        Q10: Is this scene where the main conflict reaches its peak and begins to resolve?,\n        Q11: Does this scene provide a resolution to the central narrative or thematic elements of the story?,\n        Q12: Is this scene the final, decisive moment that concludes the main plot arc?',
        # 'c_a': 'A9:No\nA10:No\nA11:No\nA12:No'}]