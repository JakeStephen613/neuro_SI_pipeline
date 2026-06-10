# Dataset utility functions
import os
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_metric, interleave_datasets


MLM_TASKS = ['bookcorpus', 'wikipedia'] # ['cc_news', 'bookcorpus', 'wikipedia', 'openwebtext', 'c4']
GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

VALIDATION_SPLIT_PERCENTAGE = 5


def load_glue_data():
    for task in GLUE_TASKS:
        load_dataset("glue", task, cache_dir='./utils/GLUE_DATA/')
        load_metric("glue", task, cache_dir='./utils/GLUE_DATA/')


def load_mlm_data():
    for task in MLM_TASKS:
        if task in ['cc_news', 'bookcorpus', 'openwebtext']:
            load_dataset(task, cache_dir='./utils/MLM_DATA/')
        elif task == 'wikipedia':
            load_dataset('wikipedia', '20220301.en', cache_dir='./utils/MLM_DATA/')
        else:
            load_dataset('c4', 'en', cache_dir='./utils/MLM_DATA')


def combine_mlm_data():
    datasets_eval, datasets_train = [], []

    if 'cc_news' in os.listdir('./utils/MLM_DATA/'):
        cc_news_eval = load_dataset('cc_news','plain_text', cache_dir='./utils/MLM_DATA', split=f"train[:{VALIDATION_SPLIT_PERCENTAGE}%]")
        non_text_column_names = [name for name in cc_news_eval.column_names if name != 'text']
        cc_news_eval = cc_news_eval.remove_columns(non_text_column_names)
        datasets_eval.append(cc_news_eval)

        cc_news_train = load_dataset('cc_news', 'plain_text', cache_dir='./utils/MLM_DATA', split=f"train[{VALIDATION_SPLIT_PERCENTAGE}%:]")
        cc_news_train =  cc_news_train.remove_columns(non_text_column_names)
        datasets_train.append(cc_news_train)

    if 'bookcorpus' in os.listdir('./utils/MLM_DATA/'):
        bookcorpus_eval = load_dataset('bookcorpus', 'plain_text', cache_dir='./utils/MLM_DATA', split=f"train[:{VALIDATION_SPLIT_PERCENTAGE}%]")
        datasets_eval.append(bookcorpus_eval)

        bookcorpus_train = load_dataset('bookcorpus', 'plain_text', cache_dir='./utils/MLM_DATA', split=f"train[{VALIDATION_SPLIT_PERCENTAGE}%:]")
        datasets_train.append(bookcorpus_train)

    if 'openwebtext' in os.listdir('./utils/MLM_DATA/'):
        openwebtext_eval = load_dataset('openwebtext', 'plain_text', cache_dir='./utils/MLM_DATA', split=f"train[:{VALIDATION_SPLIT_PERCENTAGE}%]")
        datasets_eval.append(openwebtext_eval)

        openwebtext_train = load_dataset('openwebtext','plain_text',cache_dir='./utils/MLM_DATA', split=f"train[{VALIDATION_SPLIT_PERCENTAGE}%:]")
        datasets_train.append(openwebtext_train)

    if 'wikipedia' in os.listdir('./utils/MLM_DATA/'):
        wikipedia_eval = load_dataset('wikipedia', '20220301.en', cache_dir='./utils/MLM_DATA', split=f"train[:{VALIDATION_SPLIT_PERCENTAGE}%]")
        wikipedia_eval = wikipedia_eval.remove_columns('title')
        datasets_eval.append(wikipedia_eval)

        wikipedia_train = load_dataset('wikipedia', '20220301.en', cache_dir='./utils/MLM_DATA', split=f"train[{VALIDATION_SPLIT_PERCENTAGE}%:]")
        wikipedia_train = wikipedia_train.remove_columns('title')
        datasets_train.append(wikipedia_train)

    if 'c4' in os.listdir('./utils/MLM_DATA/'):
        c4_eval = load_dataset('c4', 'en', cache_dir='./utils/MLM_DATA', split=f"validation")
        c4_eval = c4_eval.remove_columns('timestamp')
        c4_eval = c4_eval.remove_columns('url')
        datasets_eval.append(c4_eval)

        c4_train = load_dataset('c4', 'en', cache_dir='./utils/MLM_DATA', split=f"train")
        c4_train = c4_train.remove_columns('timestamp')
        c4_train = c4_train.remove_columns('url')
        datasets_train.append(c4_eval)
    
    combined_dataset = DatasetDict()
    combined_dataset['train'] = interleave_datasets(datasets_train)
    combined_dataset['validation'] = interleave_datasets(datasets_eval)

    combined_dataset.save_to_disk('./utils/MLM_DATA/combined_dataset/')


if __name__ == '__main__':
    main()
