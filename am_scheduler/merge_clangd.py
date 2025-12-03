import json
import os
import glob

OUTPUT_FILE = "compile_commands.json"

def merge_json_files():
    search_path = os.path.join(os.getcwd(), "build", "**", "compile_commands.json")
    files = glob.glob(search_path, recursive=True)
    
    combined_commands = []
    
    print(f"[LSP SETUP] total of {len(files)} command file")

    for fpath in files:
        if os.path.abspath(fpath) == os.path.abspath(OUTPUT_FILE):
            continue
            
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_commands.extend(data)
                    print(f" -> Unito: {fpath} ({len(data)} unità)")
        except Exception as e:
            print(f"Error {fpath}: {e}")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined_commands, f, indent=4)
    
    print(f"[LSP SETUP] Create {OUTPUT_FILE} with {len(combined_commands)} total commands")

if __name__ == "__main__":
    merge_json_files()
