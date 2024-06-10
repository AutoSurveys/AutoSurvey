# Project Name

This project is designed to generate detailed outlines and subsections on a given topic using an AI model. The generated content can be refined for better accuracy and completeness. The project uses Gradio for the GUI and can be run both with and without a graphical interface.

## Features

- Generates outlines based on a given topic.
- Writes detailed subsections for each outline section.
- Option to refine the generated content.
- Utilizes Gradio for a user-friendly graphical interface.
- Supports command-line interface for automation.

## Requirements

- Python 3.x
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

## Usage

### Command Line Interface (CLI)

To use the project via the command line, run the following command:

```sh
python main.py --topic "Your Topic" --api_key "Your_API_Key"
```

You can customize the arguments as needed:

- `--gpu`: Specify the GPU to use (default: '0').
- `--saving_path`: Directory to save the output (default: './output/').
- `--model`: Model to use (default: 'claude-3-haiku-20240307').
- `--topic`: Topic to generate content for.
- `--section_num`: Number of sections in the outline (default: 7).
- `--subsection_len`: Length of each subsection (default: 700).
- `--rag_num`: Number of references to use for RAG (default: 60).
- `--gui`: Whether to use the graphical interface (default: 0 for no, 1 for yes).
- `--api_key`: API key for the model.

### Graphical Interface (GUI)

To use the project with a graphical interface, set the `--gui` argument to 1:

```sh
python main.py --topic "Your Topic" --api_key "Your_API_Key" --gui 1
```

This will launch a Gradio interface where you can input your topic and other parameters.

## Example

Here is an example command to generate content on the topic "Artificial Intelligence":

```sh
python main.py --topic "Artificial Intelligence" --api_key "Your_API_Key"
```

The generated content will be saved in the `./output/` directory.

## Functions

### `remove_descriptions(text)`

Removes lines starting with "Description" from the given text.

### `write(topic, model, section_num, subsection_len, rag_num, refinement)`

Generates the outline and subsections for the given topic.

### `write_outline(topic, model, section_num)`

Generates the outline for the given topic.

### `write_subsection(topic, model, outline, subsection_len, rag_num, refinement)`

Generates subsections for the given outline.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TQDM](https://github.com/tqdm/tqdm) for progress bars.
- [Gradio](https://gradio.app) for the user interface.
- [OpenAI](https://www.openai.com) for the AI model.

---

Feel free to customize this README file according to your project's specific details and requirements.
