import os
import re
import threading
import numpy as np
from tqdm import trange,tqdm
import torch
from src.model import APIModel
import time
from src.utils import tokenCounter
import copy
import json
from src.database import database
from src.prompt import SUBSECTION_WRITING_PROMPT, LCE_PROMPT, CHECK_CITATION_PROMPT
from transformers import AutoModel, AutoTokenizer,  AutoModelForSequenceClassification

class subsectionWriter():
    
    def __init__(self, model:str, api_key:str, database) -> None:
        
        self.model, self.api_key = model, api_key 
        self.api_model = APIModel(self.model, self.api_key)

        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def write(self, topic, outline, rag_num = 30, subsection_len = 500, refining = True, reflection=True):
        # Get database
        parsed_outline = self.parse_outline(outline=outline)
        section_content = [[]] * len(parsed_outline['sections'])

        section_paper_texts = [[]] *  len(parsed_outline['sections'])
        
        total_ids = []
        section_references_ids = [[]] * len(parsed_outline['sections'])
        for i in range(len(parsed_outline['sections'])):
            descriptions = parsed_outline['subsection_descriptions'][i]
            for d in descriptions:
                references_ids = self.db.get_ids_from_query(d, num = rag_num, shuffle = False)
                total_ids += references_ids
                section_references_ids[i].append(references_ids)
        total_references_infos = self.db.get_paper_info_from_ids(list(set(total_ids)))
        temp_title_dic = {p['id']:p['title'] for p in total_references_infos}
        temp_abs_dic = {p['id']:p['abs'] for p in total_references_infos}

        for i in range(len(parsed_outline['sections'])):
            for references_ids in section_references_ids[i]:
                
                references_titles = [temp_title_dic[_] for _ in references_ids]
                references_papers = [temp_abs_dic[_] for _ in references_ids]
                paper_texts = '' 
                for t, p in zip(references_titles, references_papers):
                    paper_texts += f'---\n\npaper_title: {t}\n\npaper_content:\n\n{p}\n'
                paper_texts+='---\n'
    
                section_paper_texts[i].append(paper_texts)

        thread_l = []
        for i in range(len(parsed_outline['sections'])):
            thread = threading.Thread(target=self.write_subsection_with_reflection, args=(section_paper_texts[i], topic, outline, parsed_outline['sections'][i], parsed_outline['subsections'][i], parsed_outline['subsection_descriptions'][i], section_content, i, rag_num,str(subsection_len)))
            thread_l.append(thread)
            thread.start()
            time.sleep(0.1)
        for thread in thread_l:
            thread.join()
        raw_survey = self.generate_document(parsed_outline, section_content)
        raw_survey_with_references, raw_references = self.process_references(raw_survey)
        if refining:
            final_section_content = self.refine_subsections(topic, outline, section_content)
            refined_survey = self.generate_document(parsed_outline, final_section_content)
            refined_survey_with_references, refined_references = self.process_references(refined_survey)
            return raw_survey+'\n', raw_survey_with_references+'\n', raw_references, refined_survey+'\n', refined_survey_with_references+'\n', refined_references#, mindmap
        else:
            return raw_survey+'\n', raw_survey_with_references+'\n', raw_references#, mindmap

    def compute_price(self):
        return self.token_counter.compute_price(input_tokens=self.input_token_usage, output_tokens=self.output_token_usage, model=self.model)

    def refine_subsections(self, topic, outline, section_content):
        section_content_even = copy.deepcopy(section_content)
        
        thread_l = []
        for i in range(len(section_content)):
            for j in range(len(section_content[i])):
                if j % 2 == 0:
                    if j == 0:
                        contents = [''] + section_content[i][:2]
                    elif j == (len(section_content[i]) - 1):
                        contents = section_content[i][-2:] + ['']  
                    else:
                        contents = section_content[i][j-1:j+2]
                    thread = threading.Thread(target=self.lce, args=(topic, outline, contents, section_content_even[i], j))
                    thread_l.append(thread)
                    thread.start()
        for thread in thread_l:
            thread.join()


        final_section_content = copy.deepcopy(section_content_even)

        thread_l = []
        for i in range(len(section_content_even)):
            for j in range(len(section_content_even[i])):
                if j % 2 == 1:
                    if j == (len(section_content_even[i]) - 1):
                        contents = section_content_even[i][-2:] + ['']  
                    else:
                        contents = section_content_even[i][j-1:j+2]
                    thread = threading.Thread(target=self.lce, args=(topic, outline, contents, final_section_content[i], j))
                    thread_l.append(thread)
                    thread.start()
        for thread in thread_l:
            thread.join()
        
        return final_section_content

    def write_subsection_with_reflection(self, paper_texts_l, topic, outline, section, subsections, subdescriptions, res_l, idx, rag_num = 20, subsection_len = 1000, citation_num = 8):
        
        prompts = []
        for j in range(len(subsections)):
            subsection = subsections[j]
            description = subdescriptions[j]

            prompt = self.__generate_prompt(SUBSECTION_WRITING_PROMPT, paras={'OVERALL OUTLINE': outline,'SUBSECTION NAME': subsection,\
                                                                          'DESCRIPTION':description,'TOPIC':topic,'PAPER LIST':paper_texts_l[j], 'SECTION NAME':section, 'WORD NUM':str(subsection_len),\
                                                                            'CITATION NUM':str(citation_num)})
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        contents = self.api_model.batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [c.replace('<format>','').replace('</format>','') for c in contents]

        prompts = []
        for content, paper_texts in zip(contents, paper_texts_l):
            prompts.append(self.__generate_prompt(CHECK_CITATION_PROMPT, paras={'SUBSECTION': content, 'TOPIC':topic, 'PAPER LIST':paper_texts}))
        self.input_token_usage += self.token_counter.num_tokens_from_list_string(prompts)
        contents = self.api_model.batch_chat(prompts, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(contents)
        contents = [c.replace('<format>','').replace('</format>','') for c in contents]
    
        res_l[idx] = contents
        return contents
        
    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f'[{k}]', paras[k])
        return prompt
    
    def lce(self, topic, outline, contents, res_l, idx):
        '''
        You are an expert in artificial intelligence who wants to write a overall and comprehensive survey about [TOPIC].\n\
        You have created a overall outline below:\n\
        ---
        [OVERALL OUTLINE]
        ---
        <instruction>

        Now you need to help to refine one of the subsection to improve th ecoherence of your survey.

        You are provied with the content of the subsection "[SUBSECTION NAME]" along with the previous subsections and following subsections.

        Previous Subsection:
        --- 
        [PREVIOUS]
        ---

        Subsection to Refine: 
        ---
        [SUBSECTION]
        ---

        Following Subsection:
        ---
        [FOLLOWING]
        ---

        If the content of Previous Subsection is empty, it means that the subsection to refine is the first subsection.
        If the content of Following Subsection is empty, it means that the subsection to refine is the last subsection.

        Now edit the middle subsection to enhance coherence, remove redundancies, and ensure that it connects more fluidly with the previous and following subsections. 
        Please keep the essence and core information of the subsection intact. 
        </instruction>

        Directly return the refined subsection without any other informations:
        '''

        prompt = self.__generate_prompt(LCE_PROMPT, paras={'OVERALL OUTLINE': outline,'PREVIOUS': contents[0],\
                                                                          'FOLLOWING':contents[2],'TOPIC':topic,'SUBSECTION':contents[1]})
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        refined_content = self.api_model.chat(prompt, temperature=1).replace('<format>','').replace('</format>','')
        self.output_token_usage += self.token_counter.num_tokens_from_string(refined_content)
     #   print(prompt+'\n---------------------------------\n'+refined_content)
        res_l[idx] = refined_content
        return refined_content.replace('Here is the refined subsection:\n','')

    def parse_outline(self, outline):
        result = {
            "title": "",
            "sections": [],
            "section_descriptions": [],
            "subsections": [],
            "subsection_descriptions": []
        }
    
        # Split the outline into lines
        lines = outline.split('\n')
        
        for i, line in enumerate(lines):
            # Match title, sections, subsections and their descriptions
            if line.startswith('# '):
                result["title"] = line[2:].strip()
            elif line.startswith('## '):
                result["sections"].append(line[3:].strip())
                # Extract the description in the next line
                if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                    result["section_descriptions"].append(lines[i + 1].split('Description:', 1)[1].strip())
                    result["subsections"].append([])
                    result["subsection_descriptions"].append([])
            elif line.startswith('### '):
                if result["subsections"]:
                    result["subsections"][-1].append(line[4:].strip())
                    # Extract the description in the next line
                    if i + 1 < len(lines) and lines[i + 1].startswith('Description:'):
                        result["subsection_descriptions"][-1].append(lines[i + 1].split('Description:', 1)[1].strip())

        return result
    
    def parse_survey(self, survey):
        subsections, subdescriptions = [], []
        for i in range(100):
            if f'Subsection {i+1}' in outline:
                subsections.append(outline.split(f'Subsection {i+1}: ')[1].split('\n')[0])
                subdescriptions.append(outline.split(f'Description {i+1}: ')[1].split('\n')[0])
        return subsections, subdescriptions

    def process_references(self, survey):

        citations = self.extract_citations(survey)
        
        return self.replace_citations_with_numbers(citations, survey)

    def generate_document(self, parsed_outline, subsection_contents):
        document = []
        
        # Append title
        title = parsed_outline['title']
        document.append(f"# {title}\n")
        
        # Iterate over sections and their content
        for i, section in enumerate(parsed_outline['sections']):
            document.append(f"## {section}\n")
            # Append subsections and their contents
            for j, subsection in enumerate(parsed_outline['subsections'][i]):
                document.append(f"### {subsection}\n")
          #      document.append(f"{parsed_outline['subsection_descriptions'][i][j]}\n")
                # Append detailed content for each subsection
                if i < len(subsection_contents) and j < len(subsection_contents[i]):
                    document.append(subsection_contents[i][j] + "\n")
        
        return "\n".join(document)

    def process_outlines(self, section_outline, sub_outlines):
        res = ''
        survey_title, survey_sections, survey_section_descriptions = self.extract_title_sections_descriptions(outline=section_outline)
        res += f'# {survey_title}\n\n'
        for i in range(len(survey_sections)):
            section = survey_sections[i]
            res += f'## {i+1} {section}\nDescription: {survey_section_descriptions[i]}\n\n'
            subsections, subsection_descriptions = self.extract_subsections_subdescriptions(sub_outlines[i])
            for j in range(len(subsections)):
                subsection = subsections[j]
                res += f'### {i+1}.{j+1} {subsection}\nDescription: {subsection_descriptions[j]}\n\n'
        return res
    
    def generate_mindmap(self, subsection_citations, outline):
        to_remove = outline.split('\n')
        for _ in to_remove:
            if not '#' in _:
                outline = outline.replace(_,'')
        subsections = re.findall(pattern=r'### (.*?)\n', string=outline)
        for subs, _ in zip(subsections,range(len(subsections))):
            outline = outline.replace(subs, subs+'\n'+str(subsection_citations[_]))
        to_remove = re.findall(pattern=r'\](.*?)#', string=outline)
        for _ in to_remove:
            outline = outline.replace(_,'')
        return outline

    def extract_citations(self, markdown_text):
        # 正则表达式匹配方括号内的内容
        pattern = re.compile(r'\[(.*?)\]')
        matches = pattern.findall(markdown_text)
        # 分割引用，处理多引用情况，并去重
        citations = list()
        for match in matches:
            # 分割各个引用并去除空格
            parts = match.split(';')
            for part in parts:
                cit = part.strip()
                if cit not in citations:
                    citations.append(cit)
        return citations

    def replace_citations_with_numbers(self, citations, markdown_text):

        ids = self.db.get_titles_from_citations(citations)

        citation_to_ids = {citation: idx for citation, idx in zip(citations, ids)}

        paper_infos = self.db.get_paper_info_from_ids(ids)
        temp_dic = {p['id']:p['title'] for p in paper_infos}

        titles = [temp_dic[_] for _ in tqdm(ids)]

        ids_to_titles = {idx: title for idx, title in zip(ids, titles)}
        titles_to_ids = {title: idx for idx, title in ids_to_titles.items()}

        title_to_number = {title: num+1 for  num, title in enumerate(titles)}


        title_to_number = {title: num+1 for  num, title in enumerate(title_to_number.keys())}

        number_to_title = {num: title for  title, num in title_to_number.items()}
        number_to_title_sorted =  {key: number_to_title[key] for key in sorted(number_to_title)}

        def replace_match(match):

            citation_text = match.group(1)

            individual_citations = citation_text.split(';')

            numbered_citations = [str(title_to_number[ids_to_titles[citation_to_ids[citation.strip()]]]) for citation in individual_citations]

            return '[' + '; '.join(numbered_citations) + ']'
        

        updated_text = re.sub(r'\[(.*?)\]', replace_match, markdown_text)

        references_section = "\n\n## References\n\n"

        references = {num: titles_to_ids[title] for num, title in number_to_title_sorted.items()}
        for idx, title in number_to_title_sorted.items():
            t = title.replace('\n','')
            references_section += f"[{idx}] {t}\n\n"

        return updated_text + references_section, references
