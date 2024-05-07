# %%
import os
import re
import gc
import argparse
import torch
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama, HuggingFaceHub, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# torch.cuda.empty_cache()
# gc.collect()
# print(GPUtil.showUtilization())
# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = 'sk-proj-Y2nqeGoOCxTZnITAzuFdT3BlbkFJf6Rm2tmkcssXIov3PMFQ'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']="ls__93636df794f14ccba7162354d46779d8"
os.environ['LANGCHAIN_PROJECT']="LLM_Context_Effects"


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-m", "--model_name", required=False,
   help="name of the model. Choose one from gpt-3.5-turbo', 'gpt-4', 'mistral_7B', 'llama2_7B', 'llama2_13B', 'llama2_70B', 'llama3_8B', 'llama3_70B'")
ap.add_argument("-p", "--prompt_technique", required=False,
   help="Prompting technique (see <TODO>)")

args = vars(ap.parse_args())

if not args['model_name']:
    print('No model name given, by default using Mistral_7B. Please use -h option to see list of available model names to pass.')
    model_name = 'mistral_7B'
else:
    model_name = args['model_name']

results_dict_columns = {
    'country_pair': '',
    'prompt_style': 'single',
    'model_name': '',
    'temperature': '',
    'sim_score_1': '',
    'sim_score_2': '',
    'sim_diff': '',
    'p-value': ''
}

# Define the file path
file_path = '../data/results.csv'

# Check if the file exists
if os.path.isfile(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
else:
    # Create an empty DataFrame from the dictionary
    df = pd.DataFrame(columns=results_dict_columns)

# Print the DataFrame
print(df)

model_hf_repo_id_mapping = {
    'mistral_7B': "mistralai/Mistral-7B-Instruct-v0.2",
    'llama2_7B':'meta-llama/Llama-2-7b-chat-hf',
    'llama2_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B':'meta-llama/Meta-Llama-3-8B',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B'
}

model_name_type_mapping={
    'gpt-3.5-turbo': 'openai',
    'gpt-4': 'openai',
    'mistral_7B': 'open-source',
    'llama2_7B': 'open-source',
    'llama2_13B': 'open-source',
    'llama2_70B': 'open-source',
    'llama3_8B': 'open-source',
    'llama3_70B': 'open-source',
}

def initialise_openai_models(model_name, temperature):
    model = ChatOpenAI(model=model_name, api_key=os.environ["OPENAI_API_KEY"], temperature=temperature)
    return model

def initialise_open_source_models_transformers(model_name, temperature):
    # Use a pipeline as a high-level helper
    repo_id = model_hf_repo_id_mapping[model_name]
    pipe = pipeline("text-generation",
                    model=repo_id,
                    token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                    device_map = "balanced", max_new_tokens = 10,
                    return_full_text= False)
    return  HuggingFacePipeline(pipeline=pipe)

def initialise_open_source_models(model_name, temperature):
    repo_id = model_hf_repo_id_mapping[model_name]
    model = HuggingFaceHub(    
        repo_id=repo_id,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 10,
            "temperature": temperature,
            "device_map": 'balanced',
            'include_prompt_in_result': False
        },
        )
    # model = Ollama(model_name)
    return model


def initialise_models(model_name = 'mistral_7B', model_type = 'openai', temperature= 0.0):
    if model_type == 'openai':
        return initialise_openai_models(model_name, temperature)
    else:
        return initialise_open_source_models_transformers(model_name, temperature)


temperatures = [0.001, 0.5, 1.0, 1.5]
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

questions_order_1 = {}
questions_order_2 = {}

for country1, country2 in similarity_effect_country_pairs:
    order_1 = f'{country1}-{country2}'
    order_2 = f'{country2}-{country1}'
    questions_order_1[order_1] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country1} and {country2}? Return only a number"
    questions_order_2[order_2] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country2} and {country1}? Return only a number"

print(questions_order_1)



template = """Answer the following question in as few words as possible. 
If the question specifies the options to choose from, only output that option and no other word or token.
Your response should only be a number between 0-20 and nothing else.
Text: {text}
"""

# %%
models = ['mistral_7B',
 'llama2_7B',
 'llama2_13B',
 'llama2_70B',
 'llama3_8B',
 'llama3_70B']

def parse_numeric_output(raw_output):
    match = re.search(r'\d+', raw_output)
    # print(match, match.group())
    if match:
        return match.group()
    return None


prompt = ChatPromptTemplate.from_template(template)
model_type = model_name_type_mapping[model_name]
results_dict_columns['model_name'] = model_name
for temperature in temperatures[:1]:
        model = initialise_models(model_name, model_type, temperature)
        results_dict_columns['temperature'] = temperature
        for order_1, order_2 in zip(questions_order_1, questions_order_2):
            results_dict_columns['country_pair'] = order_1
            

            ques_1 = questions_order_1[order_1]
            ques_2 = questions_order_2[order_2]
            chain_1 = prompt | model | StrOutputParser().with_config({
        "metadata": {
                'country-pair-order': order_1,
                'model_name':model_name,
                'temperature': temperature,
            }}
            )
            chain_2 = prompt | model | StrOutputParser().with_config({
        "metadata": {
                'country-pair-order': order_2,
                'model_name':model_name,
                'temperature': temperature,
            }}
            )
            output_1 = chain_1.invoke({"text": ques_1})
            output_2 = chain_2.invoke({"text": ques_2})
            # output = chain.invoke({"text": order_2})
            parsed_output_1 = parse_numeric_output(output_1)
            if  parsed_output_1:
                    sim_score_1 =int(parsed_output_1)
            else:
                sim_score_1 = None
                print(f' cannot parse output {output_1} for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}')
            parsed_output_2 = parse_numeric_output(output_2)
            if parsed_output_2:
                sim_score_2 =int(parsed_output_2)
            else:
                print(f' cannot parse output {output_2} for Model_name: {model_name}, Pair: {order_2}, Temperature: {temperature}')
                sim_score_2 = None
            results_dict_columns['sim_score_1'] = sim_score_1
            results_dict_columns['sim_score_2'] = sim_score_2

            if sim_score_1 != None and sim_score_2 != None:
                results_dict_columns['sim_diff'] = sim_score_1 - sim_score_2
            else:
                results_dict_columns['sim_diff'] = None
            print(f'Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}, output1: {parsed_output_1}, output2: {parsed_output_2}')
            print(results_dict_columns)
            df = pd.concat([df, pd.DataFrame.from_dict([results_dict_columns])])

                
df.to_csv(file_path, index=False, mode='w')

