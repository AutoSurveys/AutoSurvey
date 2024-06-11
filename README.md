# AutoSurvey
<p align="center">
   AutoSurvey: Large Language Models Can Automatically Write Surveys
</p>
<p align="center">
  <strong></strong><br>
  Yidong Wang<sup>1,2∗</sup>, Qi Guo<sup>2,3∗</sup>, Wenjin Yao<sup>2</sup>, Hongbo Zhang<sup>1</sup>, Xin Zhang<sup>4</sup>, Zhen Wu<sup>3</sup>, Meishan Zhang<sup>4</sup>, Xinyu Dai<sup>3</sup>, Min Zhang<sup>4</sup>, Qingsong Wen<sup>5</sup>, Wei Ye<sup>2†</sup>, Shikun Zhang<sup>2†</sup>, Yue Zhang<sup>1†</sup>
  <br><br>
  <strong></strong><br>
  <sup>1</sup>Westlake University, <sup>2</sup>Peking University, <sup>3</sup>Nanjing University, <sup>4</sup>Harbin Institute of Technology, Shenzhen, <sup>5</sup>Squirrel AI
</p>

![Overview](figs/survey_pic.pdf)

## Introduction

AutoSurvey is a speedy and well-organized framework for automating the creation of comprehensive literature surveys.

## Requirements

- Python 3.10.x
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the database:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Generation
Here is an example command to generate content on the topic "In-context Learning":

```sh
python main.py --topic "In-context Learning" 
               --gpu 0
               --saving_path ./output/
               --model claude-3-haiku-20240307
               --section_num 7
               --subsection_len 700
               --rag_num 60
               --gui 0
               --api_key sk-xxxxxx 
```

The generated content will be saved in the `./output/` directory.

You can customize the arguments as needed:

- `--gpu`: Specify the GPU to use (default: '0').
- `--saving_path`: Directory to save the output survey (default: './output/').
- `--model`: Model to use (default: 'claude-3-haiku-20240307').
- `--topic`: Topic to generate content for.
- `--section_num`: Number of sections in the outline (default: 7).
- `--subsection_len`: Length of each subsection (default: 700).
- `--rag_num`: Number of references to use for RAG (default: 60).
- `--gui`: Whether to use the graphical interface (default: 0 for no, 1 for yes).
- `--api_key`: API key for the model.
- `--api_url`: API key for the model.

### Evaluation

Here is an example command to evaluate the generated survey on the topic "In-context Learning":

```sh
python main.py --topic "In-context Learning" 
               --gpu 0
               --saving_path ./output/
               --model claude-3-haiku-20240307
               --section_num 7
               --subsection_len 700
               --rag_num 60
               --gui 0
               --api_key sk-xxxxxx 
```

Make sure the generated survey is in the `./output/` directory

The evaluation result will be saved in the `./output/` directory.

- `--gpu`: Specify the GPU to use (default: '0').
- `--saving_path`: Directory to save the evaluation results (default: './output/').
- `--model`: Model to evaluate (default: 'claude-3-haiku-20240307').
- `--topic`: Topic to generate content for.
- `--api_key`: API key for the model.
- `--api_url`: API key for the model.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
