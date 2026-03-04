import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfHubHTTPError

def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model-path", type=str, required=True, help="Path to local model directory")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID on Hugging Face (e.g., username/model-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Token (optional if logged in via CLI)")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return

    print(f"Preparing to upload '{args.model_path}' to '{args.repo_id}'...")
    
    api = HfApi(token=token)
    
    # Create repo if not exists
    try:
        print(f"Checking/Creating repository '{args.repo_id}'...")
        create_repo(repo_id=args.repo_id, private=args.private, token=token, exist_ok=True)
        print(f"Repository ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Upload folder
    try:
        print("Starting upload (this may take a while for large models)...")
        upload_folder(
            folder_path=args.model_path,
            repo_id=args.repo_id,
            repo_type="model",
            token=token
        )
        print(f"Successfully uploaded model to: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Error during upload: {e}")

if __name__ == "__main__":
    main()
