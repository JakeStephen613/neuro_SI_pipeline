# instructions

## set up environment
run these in terminal:
```bash
module purge
module load anaconda3/2024.6
conda create -n graphrag python=3.11
conda activate graphrag ## just do this one from now on 
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.7.3
pip install poetry
poetry install

```


## set up data dir
- create your data directory as ```root_dir```
- copy `settings.yaml` and `.env` file from llm_kg folder into your data directory
- change ```root_dir``` in `graphrag_index_qwen.py` to your data directory
- create `input` and `output` folder in ```root_dir```
- put your text data (`.txt` files) into `input` folder

## run code
- read and modify the code
- modify the prompt and examples
- change backend LLM model
- change saving paths
- ......

run
```bash
cd ~/graphrag-clean
python graphrag_index_qwen.py
``` 
with each pipeline step