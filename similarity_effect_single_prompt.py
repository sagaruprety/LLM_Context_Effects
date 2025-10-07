# %%
import os
import re
import gc
import sys
import itertools
import json
import torch
import pandas as pd
from pydantic import BaseModel, Field
from getpass import getpass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
torch.cuda.empty_cache()
gc.collect()
# print(GPUtil.showUtilization())
# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = 'sk-proj-Y2nqeGoOCxTZnITAzuFdT3BlbkFJf6Rm2tmkcssXIov3PMFQ'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']="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"
os.environ['LANGCHAIN_PROJECT']="LLM_Context_Effects"

# model_hf_repo_id_mapping = {
#     'mistral_7B': "mistralai/Mistral-7B-Instruct-v0.2",
#     'llama2_7B':'meta-llama/Llama-2-7b-chat-hf',
#     'llama2_13B': 'meta-llama/Llama-2-13b-chat-hf',
#     'llama2_70B': 'meta-llama/Llama-2-70b-chat-hf',
#     'llama3_8B':'meta-llama/Meta-Llama-3-8B-Instruct',
#     'llama3_70B': 'meta-llama/Meta-Llama-3-70B-Instruct'
# }
models = [ 
  'llama3.2_3B',
  'llama3.1_8B',
  'llama3.1_70B',
]

model_ollama_id_mapping = {
    'llama3.2_3B':'llama3.2:3b-instruct-fp16',
    'llama3.1_8B':'llama3.1:8b-instruct-fp16',
    'llama3.1_70B': 'llama3.1:70b-instruct-fp16'
}

model_name_type_mapping={
    'gpt-4o-mini': 'openai',
    'gpt-4o': 'openai',
    'llama3.2_3B': 'open-source',
    'llama3.1_8B': 'open-source',
    'llama3.1_70B': 'open-source',
}
class CustomOutputParser(BaseModel):
    score: int = Field(description="The similarity score between 0 and 20")

def initialise_openai_models(model_name, temperature):
    model = ChatOpenAI(model=model_name, api_key=os.environ["OPENAI_API_KEY"], temperature=temperature, max_tokens=20)
    return model.with_structured_output(CustomOutputParser)

# def initialise_open_source_models_transformers(model_name, temperature):
#     # Use a pipeline as a high-level helper
#     repo_id = model_hf_repo_id_mapping[model_name]
#     pipe = pipeline("text-generation",
#                     model=repo_id,
#                     token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
#                     device_map = "sequential", max_new_tokens = 10,
#                     do_sample = True,
#                     return_full_text = False,
#                     temperature = temperature,
#                     top_k = 50,
#                     top_p = 0.9)
#     return  HuggingFacePipeline(pipeline=pipe)

def initialise_open_source_models_ollama(model_name, temperature):
    ollama_id = model_ollama_id_mapping[model_name]
    model = ChatOllama(base_url='http://localhost:11434',
    model=ollama_id, temperature = temperature, num_predict = 200, format = 'json', num_gpu=-1)
    # print(model)
    return model


def initialise_models(model_name = 'gpt-4o', model_type = 'openai', temperature= 0.0):
    if model_type == 'openai':
        return initialise_openai_models(model_name, temperature)
    else:
        return initialise_open_source_models_ollama(model_name, temperature)


# %%
temperatures = [0.8, 1.0, 1.2, 1.5]
similarity_effect_country_pairs = [
('U.S.A.', 'Mexico'),
('U.S.S.R.', 'Poland'),
('China', 'Albania'),
('U.S.A.', 'Israel'),
('Japan', 'Philippines'),
('U.S.A.', 'Canada'),
('U.S.S.R.', 'Israel'),
('England', 'Ireland'),
('Germany', 'Austria'),
('U.S.S.R.', 'France'),
('Belgium', 'Luxembourg'),
('U.S.A.', 'U.S.S.R.'),
('China', 'North Korea'),
('India', 'Sri Lanka'),
('U.S.A.', 'France'),
('U.S.S.R.', 'Cuba'),
('England', 'Jordan'),
('France', 'Israel'),
('U.S.A.', 'Germany'),
('U.S.S.R.', 'Syria'),
('France', 'Algeria')]

questions_order_and_1 = {}
questions_order_and_2 = {}

questions_order_similar_to_1 = {}
questions_order_similar_to_2 = {}

questions_order_similar_degree_1 = {}
questions_order_similar_degree_2 = {}


for country1, country2 in similarity_effect_country_pairs:
    order_1 = f'{country1}-{country2}'
    order_2 = f'{country2}-{country1}'
    questions_order_and_1[order_1] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country1} and {country2}? Shape: score: int"
    questions_order_and_2[order_2] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country2} and {country1}? Shape: score: int"
    questions_order_similar_to_1[order_1] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, is {country1} similar to {country2}? Shape: score: int"
    questions_order_similar_to_2[order_2] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, is {country2} similar to {country1}? Shape: score: int"
    questions_order_similar_degree_1[order_1] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, assess the degree to which {country1} similar to {country2}? Shape: score: int"
    questions_order_similar_degree_2[order_2] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, assess the degree to which {country2} similar to {country1}? Shape: score: int"



# %%
template = """Answer the following question to the best of your knowledge. Your answer should be a json of shape provided. 
Text: {text}
"""
modified_template = """Answer the following question to the best of your knowledge. Your answer should be a json of shape provided.
                                Text: {text}
                                Please provide an integer score.
                     """
template_cot = """Answer the following question to the best of your knowledge. Your answer should be a json of shape provided. Also, mention how you arrived at the score.
Text: {text}
Lets think step by step.
"""
prompt = ChatPromptTemplate.from_template(template)
modified_prompt = ChatPromptTemplate.from_template(modified_template)
prompt_cot = ChatPromptTemplate.from_template(template_cot)


# models = [ 'gpt-4o', 'gpt-4o-mini', 'llama3_8B']
# models = ['llama3_8B']
num_trials = 77

# %%
def parse_numeric_output(raw_output, model_type):
    if model_type == 'openai':
        return raw_output.score
    raw_output_str = str(raw_output)
    match = re.search(r'\d+', raw_output_str)
    if match:
            return int(match.group())
    return None

# %%
results_dict_columns = {
    'trial_num': [],
    'country_pair': '',
    'model_name': '',
    'temperature': '',
    'sim_score_1_to': [],
    'sim_score_2_to': [],
    'sim_score_1_and': [],
    'sim_score_2_and': [],
    'sim_score_1_degree': [],
    'sim_score_2_degree': [],
    'sim_diff_to': [],
    'sim_diff_and': [],
    'sim_diff_degree': []
}
# Define the file path
file_path = './results_sampling_exp/results_single_prompt.csv'

# Check if the file exists
if os.path.isfile(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
else:
    # Create an empty DataFrame from the dictionary
    df = pd.DataFrame(columns=results_dict_columns)


def create_chain(prompt, model, order, model_name, model_type, temperature, prompt_style):
    if model_type == 'openai':
        prompt_chain = prompt | model
    else:
        prompt_chain = prompt | model | JsonOutputParser()
    return (prompt_chain).with_config({
"metadata": {
    'country-pair-order': order,
    'model_name':model_name,
    'temperature': temperature,
    'prompt_style': prompt_style
}}
)

def get_output(prompt, model, order, model_name, temperature, ques, model_type, prompt_style):
    chain = create_chain(prompt, model, order, model_name, model_type, temperature, prompt_style)
    try:
        output = chain.invoke({"text": ques})
        return output
    except Exception as e:
        print(f"Error occurred while invoking the chain: {str(e)}")
        if "[Errno 111] Connection refused" in str(e):
            print("Connection refused. Exiting program.")
            sys.exit(1)
        return None

experimental_combinations = list(itertools.product(
    models,
    temperatures,
    similarity_effect_country_pairs,
    range(1, num_trials + 1)
))
print(len(experimental_combinations))

# Function to save progress
def save_progress(index):
    with open('experiment_progress.json', 'w') as f:
        json.dump({'last_completed_index': index}, f)

# Function to load progress
def load_progress():
    try:
        with open('experiment_progress.json', 'r') as f:
            data = json.load(f)
            return data.get('last_completed_index', -1)
    except FileNotFoundError:
        return 

# Load the last completed index
start_index = load_progress() + 1

try:
    for i, (model_name, temperature, (country1, country2), trial_num) in enumerate(experimental_combinations[start_index:], start=start_index):
        model_type = model_name_type_mapping[model_name]
        model = initialise_models(model_name, model_type, temperature)

        order_1 = f'{country1}-{country2}'
        order_2 = f'{country2}-{country1}'

        print(f'Processing combination {i+1}/{len(experimental_combinations)}: '
              f'Model: {model_name}, Temperature: {temperature}, '
              f'Countries: {order_1}, Trial: {trial_num}')

        ques_1_to = questions_order_similar_to_1[order_1]
        ques_2_to = questions_order_similar_to_2[order_2]
        ques_1_and = questions_order_and_1[order_1]
        ques_2_and = questions_order_and_2[order_2]
        ques_1_degree = questions_order_similar_degree_1[order_1]
        ques_2_degree = questions_order_similar_degree_2[order_2]


        output_1_to = get_output(prompt, model, order_1, model_name, temperature, ques_1_to, model_type, 'sst')
        output_2_to = get_output(prompt, model, order_2, model_name, temperature, ques_2_to, model_type, 'sst')
        if output_1_to is None or output_2_to is None:
            print(f"Skipping due to None output for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}")
            continue
        sim_score_1_to = parse_numeric_output(output_1_to, model_type)
        sim_score_2_to = parse_numeric_output(output_2_to, model_type)

        output_1_and = get_output(prompt, model, order_1, model_name, temperature, ques_1_and, model_type, 'ssa')
        output_2_and = get_output(prompt, model, order_2, model_name, temperature, ques_2_and, model_type, 'ssa')
        if output_1_and is None or output_2_and is None:
            print(f"Skipping due to None output for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}")
            continue
        sim_score_1_and = parse_numeric_output(output_1_and, model_type)
        sim_score_2_and = parse_numeric_output(output_2_and, model_type)

        output_1_degree = get_output(prompt, model, order_1, model_name, temperature, ques_1_degree, model_type, 'ssd')
        output_2_degree = get_output(prompt, model, order_2, model_name, temperature, ques_2_degree, model_type, 'ssd')
        if output_1_degree is None or output_2_degree is None:
            print(f"Skipping due to None output for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}")
            continue
        sim_score_1_degree = parse_numeric_output(output_1_degree, model_type)
        sim_score_2_degree = parse_numeric_output(output_2_degree, model_type)            

        if sim_score_1_to!=None and sim_score_2_to!=None:
            sim_diff_to = sim_score_1_to - sim_score_2_to
        else:
            print(f' cannot parse output {output_1_to} or {output_2_to} for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}')
            sim_diff_to = None

        if sim_score_1_and!=None and sim_score_2_and!=None:
            sim_diff_and = sim_score_1_and - sim_score_2_and
        else:
            print(f' cannot parse output {output_1_and} or {output_2_and} for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}')
            sim_diff_and = None

        if sim_score_1_degree!=None and sim_score_2_degree!=None:
            sim_diff_degree = sim_score_1_degree - sim_score_2_degree
        else:   
            print(f' cannot parse output {output_1_degree} or {output_2_degree} for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, Trial: {trial_num}, exp_index: {i}')
            sim_diff_degree = None

        results_dict_columns['temperature'] = temperature

        results_dict_columns['model_name'] = model_name
        results_dict_columns['country_pair'] = order_1
        results_dict_columns['sim_score_1_to'] = sim_score_1_to
        results_dict_columns['sim_score_2_to'] = sim_score_2_to
        results_dict_columns['sim_score_1_and'] = sim_score_1_and
        results_dict_columns['sim_score_2_and'] = sim_score_2_and
        results_dict_columns['sim_score_1_degree'] = sim_score_1_degree
        results_dict_columns['sim_score_2_degree'] = sim_score_2_degree
        results_dict_columns['sim_diff_to'] = sim_diff_to
        results_dict_columns['sim_diff_and'] = sim_diff_and
        results_dict_columns['sim_diff_degree'] = sim_diff_degree
        results_dict_columns['trial_num'] = trial_num
        # create a dataframe from the results_dict_columns
        df = pd.DataFrame.from_dict([results_dict_columns])
        
        # Write to CSV file
        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False, mode='w')
        else:
            df.to_csv(file_path, index=False, mode='a', header=False)
        save_progress(i)

except KeyboardInterrupt:
    print("\nExperiment interrupted. Progress saved.")

finally:
    # Clean up resources
    if 'model' in locals():
        print(model)
        del model
        print('Model deleted.')
        gc.collect()
        torch.cuda.empty_cache()

print("Experiment completed or interrupted. You can resume from the last saved point.")


