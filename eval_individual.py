import os
import argparse
import torch
from features import get_image_features_path, get_text_features_path
from train import get_hyperparams_str, get_save_dir

RESULT_DIR = "./results_individual/"  # Use results_individual for storing evaluation results

def get_eval_dir(args):
    """Generate the directory structure for results_individual."""
    return os.path.join(
        RESULT_DIR,
        f"{args.dataset}-shot_{args.train_shot}-seed_{args.seed}",
        f"{args.clip_encoder}",
        f"text_{args.text_layer_idx}_{args.text_augmentation}-image_{args.image_layer_idx}_{args.image_augmentation}",
        f"{args.mode}_{args.classifier_init}",
        f"logit_{args.logit}"
    )

def main(args):
    eval_dir = get_eval_dir(args)
    os.makedirs(eval_dir, exist_ok=True)

    # Check for existing results
    result_file_path = os.path.join(eval_dir, "results.csv")
    if os.path.exists(result_file_path):
        print(f"Results already exist for this experiment at {result_file_path}")
        return

    print(f"Evaluating for dataset={args.dataset}, shots={args.train_shot}, seed={args.seed}")
    
    # Load pre-extracted features
    try:
        text_features_path = get_text_features_path(
            args.dataset, args.feature_dir, args.clip_encoder, args.text_layer_idx, args.text_augmentation
        )
        text_features = torch.load(text_features_path)

        image_features_path = get_image_features_path(
            args.dataset, args.train_shot, args.seed, args.feature_dir, args.clip_encoder, args.image_layer_idx, args.image_augmentation
        )
        image_features = torch.load(image_features_path)
    except FileNotFoundError as e:
        print(f"Error: Missing features for {args.dataset}. Ensure feature extraction has been completed.")
        return

    # Dummy evaluation logic (replace with actual evaluation process)
    results = {
        "dataset": args.dataset,
        "shots": args.train_shot,
        "seed": args.seed,
        "val_acc": 0.85,  # Placeholder value
        "test_acc": 0.82  # Placeholder value
    }

    # Save results
    with open(result_file_path, "w") as f:
        f.write("dataset,shots,seed,val_acc,test_acc\n")
        f.write(f"{results['dataset']},{results['shots']},{results['seed']},{results['val_acc']},{results['test_acc']}\n")
    
    print(f"Results saved to {result_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--train-shot", type=int, required=True, help="Number of shots")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--clip-encoder", type=str, required=True, help="CLIP encoder type")
    parser.add_argument("--mode", type=str, required=True, help="Mode of training (e.g., linear)")
    parser.add_argument("--classifier_init", type=str, required=True, help="Classifier initialization (zeroshot/random)")
    parser.add_argument("--logit", type=float, required=True, help="Logit scaling parameter")
    parser.add_argument("--text_layer_idx", type=int, default=0, help="Text layer index (default is 0)")
    parser.add_argument("--text_augmentation", type=str, required=True, help="Text augmentation type")
    parser.add_argument("--image_layer_idx", type=int, default=0, help="Image layer index (default is 0)")
    parser.add_argument("--image_augmentation", type=str, required=True, help="Image augmentation type")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory for pre-extracted features")
    args = parser.parse_args()

    args = parser.parse_args()
    with torch.no_grad():
        main(args)
