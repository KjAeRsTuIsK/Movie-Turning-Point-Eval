# Movie-Turning-Point-Eval
This repo contains code for evaluating SOTA LLMs for Turning Point Identification on the TRIPOD Dataset, both for plot synopsis and full screenplay. Report is available [here](/data1/kartik/TRIPOD/Movie-Turning-Point-Eval/report.pdf).



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

We format the [TRIPOD test set](https://github.com/ppapalampidi/TRIPOD) to make it easier to run with LLMs. All the data for plot synopsis and full screenplay is available [here](/data1/kartik/TRIPOD/Movie-Turning-Point-Eval/data). Download the original test set files from [here](https://github.com/ppapalampidi/TRIPOD).

## Plot Synopsis Evaluation

For plot synopsis, we evaluate a bunch of LLMs in 0,1,5-Shot and fine-tuning setting. Our prompts are available in the [prompts.txt file](Movie-Turning-Point-Eval/data/prompts.txt)


To run a sample inference with Llama-3.1-8B for 0-Shot evaluation:

```bash
# Run the fine-tuning script
python your_script.py path/to/scene_summaries.json path/to/screenplays.csv path/to/synopses.csv path/to/output.json
```

This code will return a json file in this format:
```json
{
    "movie_name": "movie name",
    "answer": "output from llm"
}
```

For evaluation run the evaluation file:
```bash
python evaluation.py 
```

We use Unsloth to finetune Llama-3.1-8B. The reasoning instruction tuning dataset generated using Llama-3.1-70B is available [here](/data1/kartik/TRIPOD/Movie-Turning-Point-Eval/data/plot_synopsis/reasoning_instruction_dataset.json).

```bash
python your_script.py   --model_name "unsloth/Meta-Llama-3.1-8B-Instruct"\
                        --data_file "/path/to/your/data.json"\
                        --output_dir "/path/to/save/model"
```




