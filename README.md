# Movie-Turning-Point-Eval
This repo contains code for evaluating SOTA LLMs for Turning Point Identification on the TRIPOD Dataset, both for plot synopsis and full screenplay. Report is available [here](report.pdf).



## Install

1. Clone this repository and navigate to GeoChat folder
```bash
git clone https://github.com/KjAeRsTuIsK/Movie-Turning-Point-Eval.git
cd Movie-Turning-Point-Eval
```

2. Create Environment
```Shell
conda create -n tp_iden python=3.10 -y
conda activate tp_iden
pip install --upgrade pip  # enable PEP 660 support
```

## Data

We format the [TRIPOD test set](https://github.com/ppapalampidi/TRIPOD) to make it easier to run with LLMs. All the data for plot synopsis and full screenplay is available [here](data). Download the original test set files from [here](https://github.com/ppapalampidi/TRIPOD).

## Plot Synopsis 

For plot synopsis, we evaluate a bunch of LLMs in 0,1,5-Shot and fine-tuning setting. Our prompts are available in the [prompts.txt file](data/prompts.txt)


To run a sample inference with Llama-3.1-8B for 0-Shot evaluation:

```bash
python inference_llama_plot_synopsis.py /path/to/separate_sentences.json \
                 /path/to/screenplays.csv \
                 /path/to/output.json

```

This code will return a json file in this format:

```json
{
    "movie_name": "movie name",
    "answer": "output from llm"
}
```

For evaluation on the plot synopsis, run the evaluation script:

```bash
python3 evaluation_plot_synopsis.py --gt_file /path/to/gt_plot_synopsis_csv\
                                    --input_file /path/to/llm_output.json\
                                    --separate_sentences data/plot_synopsis/separate_sentences.json
```

## Finetuning on Plot Synopsis

TRIPOD dataset contains 128 (84 individual movies) training samples, with human annotation for the plot synopsis. We create multiple instruction tuning datasets, the reasoning instruction tuning dataset generated using Llama-3.1-70B is available [here](data/plot_synopsis/reasoning_instruction_dataset.json). 

We use Unsloth to LoRA finetune Llama-3.1-8B. The reasoning . First create a new environment, following [Unsloth](https://github.com/unslothai/unsloth), then run the finetuning script as shown below. Finetuning parameters can be changed within [finetuning_unsloth.py](finetuning_unsloth.py).  


```bash
python finetuning_unsloth.py   --model_name "unsloth/Meta-Llama-3.1-8B-Instruct"\
                        --data_file "/path/to/your/data.json"\
                        --output_dir "/path/to/save/model"
```


## Screenplay

For screenplay, we first generate scene level and movie summaries to reduce context length. The summaries for the test set are available here: [scenes](data/screenplay/scene_summaries.json),[movies](data/screenplay/movie_summaries.json). The few-shot examples used for screenplay evaluation are available [here](data/screenplay/few_shot_example.json).

To run a sample inference with Llama-3.1-8B for 1-Shot evaluation for the QA+DP method, run the following code:


```bash
python inference_llama_screenplay_dp_qa.py --hf_token "your_huggingface_token" \
                 --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
                 --movie_summaries_file "/path/to/movie_summaries.json" \
                 --shot_5_qa_file "/path/to/5shot_qa.json" \
                 --all_movie_scenes_summary_file "/path/to/all_movie_scenes_summaries.json" \
                 --screenplays_file "/path/to/screenplays.csv" \
                 --output_dir "/path/to/output"
```


This will give a directory with json files for each movie. Pass this folder to [change_format.py](change_format.py) to convert the output to a confidence score format.

Run the following script to get the predicted turning points as a json file using our DP algorithm and print the final scores.

```bash
python3 evaluation_screenplay.py    --gt_file /path/to/gt_screenplay_csv\
                                    --input_folder /path/to/llm_output_changed_format_folder\
                                    --scene_summaries data/screenplay/scene_summaries.json

```

## Acknowledgement

This work was done as part of my internship at MBZUAI, with [Prof Ivan Laptev](https://www.di.ens.fr/~laptev/) and [Prof Makarand Tapaswi](https://makarandtapaswi.github.io/). We are thankful to the open source community for granting access to all the LLMs, and to [Unsloth](https://github.com/unslothai/unsloth) for the awesome open-sourced faster finetuning code for LLM's.

For any queries, please reach out to [kartikkuckreja456@gmail.com](mailto:kartikkuckreja456@gmail.com)