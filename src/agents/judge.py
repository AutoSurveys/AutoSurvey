import os
import numpy as np
import tiktoken
import re
import json
from tqdm import trange,tqdm
import time
import threading
from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import CRITERIA_BASED_JUDGING_PROMPT, ROUGH_OUTLINE_PROMPT, MERGING_OUTLINE_PROMPT, SUBSECTION_OUTLINE_PROMPT, EDIT_FINAL_OUTLINE_PROMPT, NLI_PROMPT

CRITERIA = {'Coverage':{'description':'Coverage: Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics.',\
                        'score 1':'The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.',\
                        'score 2':'The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.',\
                        'score 3':'The survey is generally comprehensive in coverage but still misses a few key points that are not fully discussed.',\
                        'score 4':'The survey covers most key areas of the topic comprehensively, with only very minor topics left out.',\
                        'score 5':'The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information.',},
            
            'Structure':{'description':'Structure: Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected.',\
                        'score 1':'The survey lacks logic, with no clear connections between sections, making it difficult to understand the overall framework.',\
                        'score 2':'The survey has weak logical flow with some content arranged in a disordered or unreasonable manner.',\
                        'score 3':'The survey has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections.',\
                        'score 4':'The survey has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts.',\
                        'score 5':'The survey is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adajecent sections smooth without redundancy.',},
            
            'Relevance':{'description':'Relevance: Relevance measures how well the content of the survey aligns with the research topic and maintain a clear focus.',\
                        'score 1':'The  content is outdated or unrelated to the field it purports to review, offering no alignment with the topic',\
                        'score 2':'The survey is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.',\
                        'score 3':'The survey is generally on topic, despite a few unrelated details.',\
                        'score 4':'The survey is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.',\
                        'score 5':'The survey is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing\
                                    to a comprehensive understanding of the topic.',}}

class Judge():
    def __init__(self, model:str, api_key:str, api_url:str, database = None) -> None:

        self.model, self.api_key, self.api_url = model, api_key, api_url 
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.db = database

        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def compute_price(self):
        return self.token_counter.compute_price(input_tokens=self.input_token_usage, output_tokens=self.output_token_usage, model=self.model)

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def criteria_based_judging(self, survey, topic, criterion):
        '''
        Here is an academic survey about the topic "[TOPIC]":
        ---
        [SURVEY]
        ---

        <instruction>
        Please evaluate this survey based on the criterion above provided below, and give a score from 1 to 5 according to the score description:
        ---
        Criterion Description: [Criterion Description]
        ---
        Score 1 Description: [Score 1 Description]
        Score 2 Description: [Score 2 Description]
        Score 3 Description: [Score 3 Description]
        Score 4 Description: [Score 4 Description]
        Score 5 Description: [Score 5 Description]
        ---
        Return the score:
        '''
        criterion_paras = CRITERIA[criterion]

        content_paras = {'TOPIC':topic,'SURVEY':survey, 'Criterion Description': criterion_paras['description'],'Score 1 Description':criterion_paras['score1'], 'Score 2 Description':criterion_paras['score2'],\
                         'Score 3 Description':criterion_paras['score3'],'Score 4 Description':criterion_paras['score4'], 'Score 5 Description':criterion_paras['score5']}
        prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_PROMPT, content_paras)
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        scores = self.api_model.chat(prompt, temperature=0),
        return scores
    
    def __criteria_based_judging(self, topic, survey, criterion, res_l, idx):
        criterion_paras = CRITERIA[criterion]
        content_paras = {'TOPIC':topic,'SURVEY':survey, 'Criterion Description': criterion_paras['description']}
        for score in range(1,6):
            content_paras[f'Score {score} Description'] = criterion_paras[f'score {score}']
        prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_PROMPT, content_paras)
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        scores = self.api_model.chat(prompt, temperature=0)
        res_l[idx] = self.extract_num(scores)
        return scores
    
    def extract_num(self, string):
        numbers = re.findall(r'\d+', string)
        if len(numbers) == 0:
            return ''
        return eval(numbers[0])

    def batch_criteria_based_judging(self, survey, topic, criteria):
        '''
        Here is an academic survey about the topic "[TOPIC]":
        ---
        [SURVEY]
        ---

        <instruction>
        Please evaluate this survey based on the criterion above provided below, and give a score from 1 to 5 according to the score description:
        ---
        Criterion Description: [Criterion Description]
        ---
        Score 1 Description: [Score 1 Description]
        Score 2 Description: [Score 2 Description]
        Score 3 Description: [Score 3 Description]
        Score 4 Description: [Score 4 Description]
        Score 5 Description: [Score 5 Description]
        ---
        Return the score without any other information:
        '''
        thread_l = []
        scores = [0] * len(criteria)
        for i in range(len(criteria)):
            thread = threading.Thread(target=self.__criteria_based_judging, args=(topic, survey, criteria[i], scores, i))
            thread_l.append(thread)
            thread.start()
        for thread in thread_l:
            thread.join()
        return scores
    
    def __nli(self, sources, claim, res_l, idx):
        content_paras = {'SOURCE':'\n'.join(sources),'CLAIM':claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)

        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)

        res = self.api_model.chat(prompt, temperature=0)

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            res_l[idx] += 0
            return 0
        
    def __relevant(self, sources, com_sources, claim, res_l, idx):
        content_paras = {'SOURCE':'\n'.join(sources),'CLAIM':claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)

        res = self.api_model.chat(prompt, temperature=0)

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            content_paras = {'SOURCE':'\n'.join(com_sources),'CLAIM':claim}
            prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
            self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
            res = self.api_model.chat(prompt, temperature=0)
            if 'yes' in res.lower():
                res_l[idx] += 0
                return 0
            else:
                res_l[idx] += 1
                return 1
      
    def citation_quality(self, survey_with_reference, references):
        survey = survey_with_reference.split('## References')[0]
        survey_sections = survey.split('###')
        citation_pattern = re.compile(r'[^.!?]*\[[^\]]+\][^.!?]*[.!?]')
        sentences = []
        for content in survey_sections:
            sentences += citation_pattern.findall(content)
        claims = []
        sources_ids = []
        for s in sentences:
            sources = re.findall(pattern=r'\[(.*?)\]', string=s)
            if len(sources) > 0:
                source_ids = set()
                for ref in sources:
                    for num in ref.split(';'):
                        number = self.extract_num(num)
                        if number != '':
                            source_ids.add(number)
                if len(source_ids) >0:
                    claims.append(re.sub(pattern=r'\[(.*?)\]', repl='',string=s))
                    sources_ids.append(list(source_ids))


        paper_infos = self.db.get_paper_info_from_ids(list(references.values()))

        ids_to_title = {p['id']:p['title'] for p in paper_infos}
        ids_to_paper = {p['id']:p['abs'] for p in paper_infos}

        index_to_paper = {int(index): ids_to_paper[idx] for index, idx in references.items()}
        index_to_titles = {int(index): ids_to_title[idx] for index, idx in references.items()}

        thread_l = []
        scores = [0] * len(claims)
        for i in range(len(claims)):
            sources = [index_to_paper[index] for index in sources_ids[i]]
            thread = threading.Thread(target=self.__nli, args=(sources, claims[i], scores, i))
            thread_l.append(thread)
            thread.start()
        for thread in tqdm(thread_l):
            thread.join()
        citation_num = 0
        thread_l = []
        precisions = [0] * len(claims)
        for j, claim, source_ids in zip(range(len(claims)), claims, sources_ids):
            citation_num += len(source_ids)
            if scores[j] == 1:
                for index in source_ids:
                    sources = [index_to_paper[index]]
                    com_sources = [index_to_paper[_] for _ in source_ids if not _ == index]
                    thread = threading.Thread(target=self.__relevant, args=(sources, com_sources, claim, precisions, j))
                    thread_l.append(thread)
                    thread.start()
        for thread in tqdm(thread_l):
            thread.join()

        precisions = np.array(precisions)

        return np.array(scores).mean(), precisions.sum()/citation_num
