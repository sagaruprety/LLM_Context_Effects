# %%
import os
import re
import gc
# import GPUtil
import torch
import pandas as pd
from getpass import getpass
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama, HuggingFaceHub, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
torch.cuda.empty_cache()
gc.collect()
# print(GPUtil.showUtilization())
# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = 'sk-proj-Y2nqeGoOCxTZnITAzuFdT3BlbkFJf6Rm2tmkcssXIov3PMFQ'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']="ls__93636df794f14ccba7162354d46779d8"
os.environ['LANGCHAIN_PROJECT']="LLM_Context_Effects"

# %%
model_hf_repo_id_mapping = {
    'mistral_7B': "mistralai/Mistral-7B-Instruct-v0.2",
    'llama2_7B':'meta-llama/Llama-2-7b-chat-hf',
    'llama2_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B':'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B-Instruct'
}

model_ollama_id_mapping = {
    'mistral_7B': "mistral:7b-instruct-fp16",
    'llama2_7B': 'llama2:7b-chat-fp16',
    'llama2_13B': 'llama2:13b-chat-fp16',
    'llama2_70B': 'llama2:70b-chat-fp16',
    'llama3_8B':'llama3:8b-instruct-fp16',
    'llama3_70B': 'llama3:70b-instruct-fp16'
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
    model = ChatOpenAI(model=model_name, api_key=os.environ["OPENAI_API_KEY"], temperature=temperature, max_tokens=10)
    return model

def initialise_open_source_models_transformers(model_name, temperature):
    # Use a pipeline as a high-level helper
    repo_id = model_hf_repo_id_mapping[model_name]
    pipe = pipeline("text-generation",
                    model=repo_id,
                    token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                    device_map = "sequential", max_new_tokens = 10,
                    do_sample = True,
                    return_full_text = False,
                    temperature = temperature,
                    top_k = 50,
                    top_p = 0.9)
    return  HuggingFacePipeline(pipeline=pipe)

def initialise_open_source_models_ollama(model_name, temperature):
    ollama_id = model_ollama_id_mapping[model_name]
    model = Ollama(base_url='http://localhost:11434',
    model=ollama_id, temperature = temperature, num_predict = 10, format = 'json', num_gpu=-1)
    print(model)
    return model


def initialise_models(model_name = 'mistral_7B', model_type = 'openai', temperature= 0.0):
    if model_type == 'openai':
        return initialise_openai_models(model_name, temperature)
    else:
        return initialise_open_source_models_ollama(model_name, temperature)


# %%
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
    questions_order_1[order_1] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country1} and {country2}? Shape: score: int"
    questions_order_2[order_2] = f"On a scale of 0 to 20, where 0 means no similarity and 20 means complete similarity, how similar are {country2} and {country1}? Shape: score: int"

# print(questions_order_1)


# %%
template = """Answer the following question to the best of your knowledge. Your answer should be a json of shape provided.
Text: {text}
"""
modified_template = """Answer the following question to the best of your knowledge. Your answer should be a json of shape provided.
                                Text: {text}
                                Please provide an integer score.
                     """
prompt = ChatPromptTemplate.from_template(template)
modified_prompt = ChatPromptTemplate.from_template(modified_template)

# %%
models = [
    'mistral_7B',
 'llama2_7B',
  'llama3_8B',
 'llama2_13B',
 'llama2_70B',
 'gpt-3.5-turbo',
 'gpt-4']
# models = [ 'mistral_7B']

# %%
def parse_numeric_output(raw_output):
    match = re.search(r'\d+', raw_output)
    # print(match, match.group())
    if match:
        return match.group()
    return None


# %%
results_dict_columns = {
    'country_pair': '',
    'prompt_style': 'single',
    'model_name': '',
    'temperature': '',
    'sim_score_1': [],
    'sim_score_2': [],
    'sim_diff': [],
    'p-values': []
}
# Define the file path
file_path = './results/results_variability.csv'

# Check if the file exists
if os.path.isfile(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
else:
    # Create an empty DataFrame from the dictionary
    df = pd.DataFrame(columns=results_dict_columns)

# Print the DataFrame
# print(df)

# %%
def create_chain(prompt, model, order, model_name, temperature):
    return prompt | model | StrOutputParser().with_config({
"metadata": {
    'country-pair-order': order,
    'model_name':model_name,
    'temperature': temperature,
}}
)

def get_output(prompt, model, order, model_name, temperature, ques):
    chain = create_chain(prompt, model, order, model_name, temperature)
    return chain.invoke({"text": ques})

# %%

for model_name in models:
    model_type = model_name_type_mapping[model_name]
    results_dict_columns['model_name'] = model_name
    for temperature in temperatures:
        model = initialise_models(model_name, model_type, temperature)
        results_dict_columns['temperature'] = temperature
        for order_1, order_2 in zip(questions_order_1, questions_order_2):
            sim_score_1_list = []
            sim_score_2_list = []
            sim_diff_list = []
            results_dict_columns['country_pair'] = order_1
            for i in range(10):
                print(f'Model - {model_name}, temp - {temperature} Iteration No. {i+1}')
                ques_1 = questions_order_1[order_1]
                ques_2 = questions_order_2[order_2]
                chain_1 = create_chain(prompt, model, order_1, model_name, temperature)    
                chain_2 = create_chain(prompt, model, order_1, model_name, temperature)
                output_1 = get_output(prompt, model, order_1, model_name, temperature, ques_1)
                output_2 = get_output(prompt, model, order_2, model_name, temperature, ques_2)
                parsed_output_1 = parse_numeric_output(output_1)
                if  parsed_output_1:
                        sim_score_1 = int(parsed_output_1)
                else:
                    output_1 = get_output(modified_prompt, model, order_1, model_name, temperature, ques_1)
                    parsed_output_1 = parse_numeric_output(output_1)
                    if  parsed_output_1:
                        sim_score_1 = int(parsed_output_1)
                    else:
                        sim_score_1 = None
                        print(f' cannot parse output {output_1} for Model_name: {model_name}, Pair: {order_1}, Temperature: {temperature}')
                sim_score_1_list.append(sim_score_1)
                parsed_output_2 = parse_numeric_output(output_2)
                if parsed_output_2:
                    sim_score_2 = int(parsed_output_2)
                    
                else:
                    output_2 = get_output(modified_prompt, model, order_2, model_name, temperature, ques_2)
                    parsed_output_2 = parse_numeric_output(output_2)
                    if  parsed_output_2:
                        sim_score_2 = int(parsed_output_2)
                    else:
                        sim_score_2 = None
                        print(f' cannot parse output {output_2} for Model_name: {model_name}, Pair: {order_2}, Temperature: {temperature}')
                sim_score_2_list.append(sim_score_2)
                if sim_score_1!=None and sim_score_2!=None:
                    sim_diff_list.append(sim_score_1 - sim_score_2)
                else:
                    sim_diff_list.append(None)
                print(f'for Model_name: {model_name}, Pair: {order_2}, Temperature: {temperature}, output1: {output_1}, output2: {output_2}')
            results_dict_columns['sim_score_1'] = sim_score_1_list
            results_dict_columns['sim_score_2'] = sim_score_2_list
            results_dict_columns['sim_diff'] = sim_diff_list
            df = pd.concat([df, pd.DataFrame.from_dict([results_dict_columns])])
            # del model
        # print('model deleted..')
        # gc.collect()
        # torch.cuda.empty_cache()
                
                

# %%
# df.head(20)

# # %%
# sim_score_1_all = df.iloc[0, 4]
# sim_score_2_all = df.iloc[0, 5]

# # %%
# len(sim_score_1_all)

# # %%
# len(df)

# %%
df.to_csv(file_path, index=False, mode='w')

# %%
# del model
# print('model deleted..')
# gc.collect()
# torch.cuda.empty_cache()

# %%


# %%



