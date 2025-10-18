import argparse
from fetch_dataset import load_existing_ids, save_ids,initialize_zip_count,open_new_zip_file

def preprocess_uma():
    
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dataset crawler for Qwen-Sovits"
    )
    parser.add_argument(
        "-o", 
        "--output_dir", 
        type=str, 
        default="Z:/sata11-18612520532/AI/TTS/dataset", 
        help="Path to save the dataset"
    )
    parser.add_argument(
        "-s", 
        "--dataset_source", 
        type=str, 
        default="Umamusume", 
        help="Dataset source"
    )
    parser.add_argument(
        "-ss", 
        "--dataset_subset", 
        type=str, 
        default="八重神子", 
        help="Dataset sub set"
    )
    parser.add_argument(
        "-l", 
        "--lang", 
        type=str, 
        default="ja", 
        help="Dataset Language"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=10*60*60, 
        help="Dataset Language"
    )
    parser.add_argument(
        '--repack',
        type=int,
        default=0,
        help='Repack the dataset to use zip file to store the files'
    )
    args = parser.parse_args()