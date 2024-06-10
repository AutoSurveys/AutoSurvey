import os
import json
import argparse
from src.agents.outline_writer import outlineWriter
from src.agents.writer import subsectionWriter
from src.agents.judge import Judge
from src.database import database
from tqdm import tqdm
import time
import gradio as gr

def remove_descriptions(text):
    lines = text.split('\n')
    
    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]
    
    result = '\n'.join(filtered_lines)
    
    return result

def write(topic, model, section_num, subsection_len, rag_num, refinement):
    outline, outline_wo_description = write_outline(topic, model, section_num)

    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = True)
        return refined_survey_with_references
    else:
        raw_survey, raw_survey_with_references, raw_references = write_subsection(topic, model, outline, subsection_len = subsection_len, rag_num = rag_num, refinement = False)
        return raw_survey_with_references

def write_outline(topic, model, section_num, db, api_key):
    outline_writer = outlineWriter(model=model, api_key=api_key, database=db)
    outline = outline_writer.draft_outline(topic, 1500, 30000, section_num)
    return outline, remove_descriptions(outline)

def write_subsection(topic, model, outline, subsection_len, rag_num, db, api_key, refinement = True):

    subsection_writer = subsectionWriter(model=model, api_key=api_key, database=db)
    if refinement:
        raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, refining = True)
        return raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references
    else:
        raw_survey, raw_survey_with_references, raw_references = subsection_writer.write(topic, outline, subsection_len = subsection_len, rag_num = rag_num, refining = False)
        return raw_survey, raw_survey_with_references, raw_references

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='')
    parser.add_argument('--saving_path',default='./output/', type=str, help='')
    parser.add_argument('--model',default='claude-3-haiku-20240307', type=str, help='')
    parser.add_argument('--topic',default='', type=str, help='')
    parser.add_argument('--section_num',default=7, type=int, help='')
    parser.add_argument('--subsection_len',default=700, type=int, help='')
    parser.add_argument('--rag_num',default=60, type=str, help='')
    parser.add_argument('--api_key',default='', type=str, help='')
    parser.add_argument('--db_path',default='../database', type=str, help='')
    parser.add_argument('--embedding_model',default='../model/nomic-embed-text-v1', type=str, help='')
    args = parser.parse_args()
    return args

def main(args):

    db = database(db_path = args.db_path, embedding_model = args.embedding_model)
    
    api_key = args.api_key

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    outline_with_description, outline_wo_description = write_outline(args.topic, args.model, args.section_num, db, args.api_key)

    raw_survey, raw_survey_with_references, raw_references, refined_survey, refined_survey_with_references, refined_references = write_subsection(args.topic, args.model, outline_with_description, args.subsection_len, args.rag_num, db, args.api_key)

    with open(f'{args.saving_path}/{args.topic}.md', 'a+') as f:
        f.write(refined_survey_with_references)
    with open(f'{args.saving_path}/{args.topic}.json', 'a+') as f:
        save_dic = {}
        save_dic['survey'] = refined_survey_with_references
        save_dic['reference'] = refined_references
        f.write(json.dumps(save_dic, indent=4))

if __name__ == '__main__':

    args = paras_args()

    main(args)