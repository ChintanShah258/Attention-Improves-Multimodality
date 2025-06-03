from copy import deepcopy
import os
import time

import torch
from torch.utils.data import DataLoader

from engine.config import parser

from engine.tools.utils import makedirs, set_random_seed
from engine.datasets.utils import TensorDataset, TextTensorDataset
from engine.model.head import make_classifier_head, get_zero_shot_weights
from engine.model.logit import LogitHead
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.optim import build_optimizer
from engine.optimizer.scheduler import build_lr_scheduler
from features import get_backbone_name, \
                     get_few_shot_setup_name, \
                     get_view_name, \
                     get_image_features_path, \
                     get_text_features_path, \
                     get_image_encoder_dir, \
                     get_text_encoder_dir, \
                     get_test_features_path

from engine.gating.gating_mechanism import GatingMechanism
from engine.gating.GatingMechanismTrainer import GatingMechanismTrainer

torch.set_num_threads(4) # To maximize efficiency, please tune the number of threads for your machine

CROSS_MODAL_BATCH_RATIO = 0.5 # Half of the batch is image, the other half is text
EVAL_FREQ = 100 # Evaluate on val set per 100 iterations (for early stopping)


def get_benchmark_name(dataset, train_shot, seed):
    benchmark_name = "-".join([
        dataset,
        get_few_shot_setup_name(train_shot, seed)
    ])
    return benchmark_name


def get_modality_name(modality,
                      clip_encoder,
                      image_augmentation,
                      text_augmentation,
                      image_layer_idx,
                      text_layer_idx,
                      image_views=1):
    text_feature_name = f"text_{text_layer_idx}_{text_augmentation}"
    image_feature_name = f"image_{image_layer_idx}_{get_view_name(image_augmentation, image_views=image_views)}"
    if modality == "cross_modal":
        feature_name = f"{text_feature_name}-{image_feature_name}"
    elif modality == "uni_modal":
        feature_name = image_feature_name
    return os.path.join(
        get_backbone_name(clip_encoder),
        feature_name
    )


def get_architecture_name(classifier_head, classifier_init):
    return classifier_head + "_" + classifier_init


def get_logit_name(logit):
    name = f"logit_{logit}"
    return name


def get_save_dir(args):
    save_dir = os.path.join(
        args.result_dir,
        get_benchmark_name(
            args.dataset,
            args.train_shot,
            args.seed
        ),
        get_modality_name(
            args.modality,
            args.clip_encoder,
            args.image_augmentation,
            args.text_augmentation,
            args.image_layer_idx,
            args.text_layer_idx,
            image_views=args.image_views
        ),
        get_architecture_name(
            args.classifier_head,
            args.classifier_init
        ),
        get_logit_name(
            args.logit
        ),
    )
    return save_dir


def get_hyperparams_str(optim,
                        lr,
                        wd,
                        batch_size,
                        iters):
    hyperparams_str = f"optim_{optim}-lr_{lr}-wd_{wd}-bs_{batch_size}-iters_{iters}"
    return hyperparams_str


def get_wiseft(head, zero_shot_weights, wiseft_ratio=0.5):
    if type(head) == torch.nn.Linear:
        head.weight.data = (1 - wiseft_ratio) * head.weight.data + wiseft_ratio * torch.nn.functional.normalize(zero_shot_weights, dim=1)
    elif type(head) == torch.nn.Sequential:
        assert type(head[-1]) == torch.nn.Linear, f"Invalid head: {head}"
        head[-1].weight.data = (1 - wiseft_ratio) * head[-1].weight.data + wiseft_ratio * torch.nn.functional.normalize(zero_shot_weights, dim=1)
    return head


def get_eval_heads(head, zero_shot_weights, ratio_list=[0.5], logit=None):
    logit_head = LogitHead(
        deepcopy(head),
        logit_scale=logit,
    )

    eval_heads = {
        'head': logit_head.cuda().eval(),
    }
    for ratio in ratio_list:
        # TODO (Warning): This wise-ft does not consider partial finetuning of image encoder
        wiseft = get_wiseft(deepcopy(head), zero_shot_weights, ratio)
        wiseft_head = LogitHead(
            wiseft,
            logit_scale=logit,
        )
        eval_heads[f'head_wiseft_{ratio}'] = wiseft_head.cuda().eval()
    return eval_heads

# Preprocess text_loader into a tensor
def preprocess_text_loader(text_loader, text_encoder, device="cuda"):
    """
    Preprocess and normalize text features from the text_loader.

    Args:
        text_loader (DataLoader): Loader providing text, labels, and eot_indices.
        text_encoder (nn.Module): Encoder to generate text features.
        device (str): Device to perform operations on (e.g., "cuda").

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Normalized text features, labels, and eot_indices.
    """
    processed_features = []
    processed_labels = []
    processed_eot_indices = []

    for text_batch, text_labels, eot_indices in text_loader:
        # Move data to the device
        text_batch = text_batch.to(device)
        text_labels = text_labels.to(device)
        eot_indices = eot_indices.to(device)

        # Pass text through the text_encoder to extract features
        text_features = text_encoder(text_batch, eot_indices)

        # Append results to lists
        processed_features.append(text_features)
        processed_labels.append(text_labels)
        processed_eot_indices.append(eot_indices)

    # Combine all batches into single tensors
    processed_features = torch.cat(processed_features, dim=0)
    processed_labels = torch.cat(processed_labels, dim=0)
    processed_eot_indices = torch.cat(processed_eot_indices, dim=0)

    # Normalize the combined text features
    processed_features = torch.nn.functional.normalize(processed_features, p=2, dim=1)

    return processed_features, processed_labels, processed_eot_indices


def train_gating_mechanism(gating_mechanism, image_loader, processed_text_features, trainer, num_epochs, device="cuda"):
    """
    Train the Gating Mechanism using image features from the provided image_loader and processed text features.

    Args:
        gating_mechanism (nn.Module): The Gating Mechanism model.
        image_loader (DataLoader): DataLoader containing image features and labels.
        processed_text_features (Tensor): Preprocessed and normalized text features.
        trainer (GatingMechanismTrainer): Trainer for the Gating Mechanism.
        num_epochs (int): Number of epochs to train the Gating Mechanism.
        device (str): Device to use for training (e.g., "cuda").

    Returns:
        None: The Gating Mechanism is updated in-place.
    """
    # Move processed_text_features to the device
    processed_text_features = processed_text_features.to(device)

    # Train for the specified number of epochs
    for epoch in range(num_epochs):
        print(f"Training Gating Mechanism - Epoch {epoch+1}/{num_epochs}")
        for image_batch, _ in image_loader:
            # Move image features to the device
            image_batch = image_batch.to(device)
            
            # Train the Gating Mechanism for this batch
            trainer.train(image_batch, processed_text_features)

    print("Gating Mechanism training complete.")

def preprocess_and_enhance_features(
    image_loader, processed_text_features, image_encoder, gating_mechanism, trainer, device="cuda", train_mode=False
):
    """
    Preprocess and enhance image features using the GatingMechanism with optional training.

    Args:
        image_loader (DataLoader): DataLoader for image features.
        processed_text_features (Tensor): Precomputed and normalized text features.
        image_encoder (nn.Module): Image encoder model.
        gating_mechanism (nn.Module): Gating mechanism model.
        trainer (GatingMechanismTrainer): Trainer for the gating mechanism.
        device (str): Device to use (e.g., 'cuda').
        train_mode (bool): Whether to train the gating mechanism.

    Returns:
        Tuple[Tensor, Tensor]: Enhanced image features and corresponding labels.
    """
    # ------------------------------------------
    # Step 1: Compute All Image Features (One-Time Encoding)
    # ------------------------------------------
    image_features = []
    image_labels = []

    for image_batch, image_label in image_loader:
        image_batch = image_batch.to(device)
        image_label = image_label.to(device)
        with torch.no_grad():
            image_feature = image_encoder(image_batch)
        image_features.append(image_feature)
        image_labels.append(image_label)

    image_features = torch.cat(image_features, dim=0)
    image_labels = torch.cat(image_labels, dim=0)

    # Normalize image features
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

    # ------------------------------------------
    # Train or Enhance Features with Gating Mechanism
    # ------------------------------------------
    if train_mode:
        enhanced_image_features = trainer.train(image_features, processed_text_features)
    else:
        gating_mechanism.eval()  # Ensure gating mechanism is in eval mode
        with torch.no_grad():
            enhanced_image_features = gating_mechanism(image_features, processed_text_features)

    # Move all data to CPU
    enhanced_image_features = enhanced_image_features.to("cpu")
    image_labels = image_labels.to("cpu")

    return enhanced_image_features, image_labels

def train(logit_head, train_enhanced_loader, processed_text_loader, val_enhanced_loader,image_encoder, text_encoder,
          optimizer, scheduler, criterion, iters, eval_freq=EVAL_FREQ, device="cuda"):
    """
    Train the logit head using enhanced (image + text) features.
    Combines enhanced image features with normalized text features before passing to the logit head.
    """
    # Initialize iterators for the data loaders
    processed_text_loader_iter = iter(processed_text_loader)
    train_enhanced_loader_iter = iter(train_enhanced_loader)

    result_dict = {
        "iter": None,
        "val_acc": None,
        "image_encoder": None,
        "text_encoder": None,
        "logit_head": None,
    }

    for i in range(iters):
        logit_head.train()  # Set logit head to training mode

        # Reset data loader iterators if needed
        try:
            enhanced_features, image_labels = next(train_enhanced_loader_iter)
        except StopIteration:
            train_enhanced_loader_iter = iter(train_enhanced_loader)
            enhanced_features, image_labels = next(train_enhanced_loader_iter)

        try:
            text_features, text_labels, _ = next(processed_text_loader_iter)
        except StopIteration:
            processed_text_loader_iter = iter(processed_text_loader)
            text_features, text_labels, _ = next(processed_text_loader_iter)

        # Move data to device
        enhanced_features = enhanced_features.to(device)  # Enhanced image features
        image_labels = image_labels.to(device)
        text_features = text_features.to(device)
        text_labels = text_labels.to(device)

        # Detach text features to prevent retaining the computational graph
        text_features = text_features.detach()
        enhanced_features = enhanced_features.detach()

        # Concatenate enhanced image features with text features
        combined_features = torch.cat([enhanced_features, text_features], dim=0)
        combined_labels = torch.cat([image_labels, text_labels], dim=0)

        # Forward pass through the logit head
        logits = logit_head(combined_features)

        # Compute loss
        loss = criterion(logits, combined_labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=False)  # Ensure no graph is retained for reused tensors
        optimizer.step()

        # Scheduler step
        scheduler.step()

        # Periodic evaluation
        if i % eval_freq == 0:
            val_acc = validate(logit_head, val_enhanced_loader, device="cuda")
            if result_dict["val_acc"] is None or val_acc > result_dict["val_acc"]:
                result_dict["iter"] = i
                result_dict["val_acc"] = val_acc
                result_dict["image_encoder"] = deepcopy(image_encoder.state_dict())
                result_dict["text_encoder"] = deepcopy(text_encoder.state_dict())
                result_dict["logit_head"] = deepcopy(logit_head.state_dict())

    # Load the best model
    logit_head.load_state_dict(result_dict["logit_head"])
    val_acc = validate(logit_head, val_enhanced_loader, device="cuda")
    print(f"Best validation accuracy: {result_dict['val_acc']:.4f} at iteration {result_dict['iter']}")
    return result_dict

def validate(logit_head, enhanced_loader, device="cuda"):
    """
    Validate the logit head using enhanced image features only.
    """
    logit_head.eval()  # Set the logit head to evaluation mode
    val_acc = 0
    val_count = 0

    with torch.no_grad():
        for enhanced_features, labels in enhanced_loader:
            # Move data to device
            enhanced_features = enhanced_features.to(device)
            labels = labels.to(device)

            # Forward pass through the logit head
            logits = logit_head(enhanced_features)

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            val_acc += torch.sum(predictions == labels).item()
            val_count += labels.size(0)

    return val_acc / val_count

def get_valid_batch_sizes(hyperparams, text_dataset, image_train_dataset, batch_ratio=CROSS_MODAL_BATCH_RATIO, modality='cross_modal'):
    VALID_BATCH_SIZES = []
    if modality == 'uni_modal':
        batch_ratio = 0.
    for batch_size in hyperparams['batch_size']:
        text_batch_size = int(batch_size * batch_ratio)
        image_batch_size = batch_size - text_batch_size
        # check if text batch size is smaller than the size of text dataset
        if text_batch_size == 0 or text_batch_size < len(text_dataset):
            # check if image batch size is smaller than the size of image dataset
            if image_batch_size == 0 or image_batch_size < len(image_train_dataset):
                VALID_BATCH_SIZES.append(batch_size)
    if len(VALID_BATCH_SIZES) == 0:
        raise ValueError("No valid batch size found. You should consider reducing the batch size.")
    print("Valid batch sizes: {}/{}".format(len(VALID_BATCH_SIZES), len(hyperparams['batch_size'])))
    return VALID_BATCH_SIZES

def main(args):
    # Initialize the Gating Mechanism
    gating_mechanism = GatingMechanism(top_k=3).to("cuda")
    # Global Flag for Freezing Gating Mechanism
    freeze_gating = False  # Set True to freeze the gating mechanism for testing/validation

    if args.seed >= 0:
        print("Setting fixed seed: {}".format(args.seed))
        set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    image_encoder_dir = get_image_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    image_encoder_path = os.path.join(image_encoder_dir, "encoder.pth")

    text_encoder_dir = get_text_encoder_dir(
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")

    text_features_path = get_text_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.text_layer_idx,
        args.text_augmentation
    )
    text_features = torch.load(text_features_path)
    # text_features['features'] = torch.nn.functional.normalize(text_features['features'], dim=1)
    text_dataset = TextTensorDataset(
        text_features['features'], text_features['labels'], text_features['eot_indices'])

    #print(f"text_dataset shape: {text_dataset.shape}")

    
    ccrop_features_path = get_image_features_path(
        args.dataset,
        args.train_shot,
        args.seed,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx,
        "none",
    )
    ccrop_features = torch.load(ccrop_features_path)

    if args.image_augmentation == "none":
        train_features = ccrop_features['train']['features']
        train_labels = ccrop_features['train']['labels']
    else:
        # Add extra views
        image_features_path = get_image_features_path(
            args.dataset,
            args.train_shot,
            args.seed,
            args.feature_dir,
            args.clip_encoder,
            args.image_layer_idx,
            args.image_augmentation,
            image_views=args.image_views,
        )
        image_features = torch.load(image_features_path)
        train_features = torch.cat([ccrop_features['train']['features'], image_features['train']['features']], dim=0)
        train_labels = torch.cat([ccrop_features['train']['labels'], image_features['train']['labels']], dim=0)
    
    image_train_dataset = TensorDataset(
        train_features,
        train_labels
    )
    
    #print(f"image_traing_dataset shape: {image_train_dataset.shape}")
    
    image_val_dataset = TensorDataset(
        ccrop_features['val']['features'],
        ccrop_features['val']['labels']
    )
    
    print(f"image_val_dataset shape: {image_val_dataset.shape}")

    test_features_path = get_test_features_path(
        args.dataset,
        args.feature_dir,
        args.clip_encoder,
        args.image_layer_idx
    )
    test_features = torch.load(test_features_path)
    test_dataset = TensorDataset(
        test_features['features'],
        test_features['labels']
    )
    
    #print(f"test_dataset shape: {test_dataset.shape}")
    
    save_dir = get_save_dir(args)
    
    hyperparams = HYPER_DICT[args.hyperparams]
    # filter out invalid batch sizes
    VALID_BATCH_SIZES = get_valid_batch_sizes(hyperparams, text_dataset, image_train_dataset, modality=args.modality)

    def get_experiment_count(hyperparams):
        count = 1
        count *= len(hyperparams['lr'])
        count *= len(hyperparams['weight_decay'])
        count *= len(VALID_BATCH_SIZES)
        count *= len(hyperparams['max_iter'])
        return count
    experiment_count = get_experiment_count(hyperparams)
    cur_count = 0
    # sweep through hyperparameters
    for lr in hyperparams['lr']:
        for wd in hyperparams['weight_decay']:
            for batch_size in VALID_BATCH_SIZES:
                for iters in hyperparams['max_iter']:
                    cur_count += 1

                    hyperparams_str = get_hyperparams_str(
                        hyperparams['optim'], lr, wd, batch_size, iters)
                    
                    # check if experiment has been done
                    checkpoint_dir = os.path.join(save_dir, hyperparams_str)
                    makedirs(checkpoint_dir)
                    test_result_dict = {}
                    test_result_path = os.path.join(checkpoint_dir, "test_result.pth")
                    if os.path.exists(test_result_path):
                        print(f"Already exists: {hyperparams_str} {cur_count}/{experiment_count}")
                        test_result_dict = torch.load(test_result_path)
                        continue
                    else:
                        print(f"Starting: {hyperparams_str} {cur_count}/{experiment_count}")
                    
                    # train logreg

                    # Create the logreg model
                    image_encoder = torch.load(
                        image_encoder_path).partial_model.train().cuda()
                    text_encoder = torch.load(
                        text_encoder_path).partial_model.train().cuda()
                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init,
                        text_dataset,
                        text_encoder
                    )
                    logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    ).train().cuda()
                    # Create the optimizer
                    params_groups = [
                        {'params': logit_head.parameters()},
                        {'params': image_encoder.parameters()},
                        {'params': text_encoder.parameters()},
                    ]
                    
                    # Include gating mechanism parameters if not frozen
                    if not freeze_gating:
                        params_groups.append({'params': gating_mechanism.parameters()})
    
                    optimizer = build_optimizer(params_groups, hyperparams['optim'], lr, wd)
                    scheduler = build_lr_scheduler(
                        optimizer,
                        hyperparams['lr_scheduler'],
                        hyperparams['warmup_iter'],
                        iters,
                        warmup_type=hyperparams['warmup_type'],
                        warmup_lr=hyperparams['warmup_min_lr']
                    )
                    criterion = torch.nn.CrossEntropyLoss()

                    if args.modality == "cross_modal":
                        text_batch_size = int(batch_size * CROSS_MODAL_BATCH_RATIO)
                    elif args.modality == "uni_modal":
                        text_batch_size = 0
                    image_batch_size = batch_size - text_batch_size

                    text_loader = None
                    if text_batch_size > 0:
                        text_loader = DataLoader(
                            text_dataset,
                            batch_size=len(text_dataset), # Load all text samples at once
                            shuffle=False, # Static text dataset. No shuffling since complete text is fed
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )
                    
                    # Preprocess text features
                    processed_text_features, processed_text_labels, processed_text_eot_indices = preprocess_text_loader(
                        text_loader=text_loader,
                        text_encoder=text_encoder,
                        device="cuda"
                        )
                    
                    # Create a new dataset with the processed text data
                    processed_text_dataset = TextTensorDataset(
                        processed_text_features, processed_text_labels, processed_text_eot_indices
                    )
                    
                    # Update the text_loader with the processed dataset
                    processed_text_loader = DataLoader(
                        processed_text_dataset,
                        batch_size=len(processed_text_dataset),  # Replace with your desired batch size
                        shuffle=True,  # Shuffle if required for training
                        num_workers=args.num_workers,
                        pin_memory=False
                    )
                    
                    image_loader = None
                    if image_batch_size > 0:
                        image_loader = DataLoader(
                            image_train_dataset,
                            batch_size=image_batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                        )
                    
                    for batch in image_loader:
                            print(batch)  # Inspect the structure of the batch
                            break

                    
                    val_loader = DataLoader(
                        image_val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    
                    # Initialize the GatingMechanismTrainer
                    gating_trainer = GatingMechanismTrainer(
                        model=gating_mechanism,
                        optimizer=torch.optim.Adam(gating_mechanism.parameters(), lr=1e-3),
                        scheduler=torch.optim.lr_scheduler.StepLR(
                            torch.optim.Adam(gating_mechanism.parameters(), lr=1e-3), step_size=5, gamma=0.5
                        ),
                        device="cuda",
                        lambda_reconstruction=0.75,
                        lambda_alignment= 0.25
                        )
                    
                    # Train the Gating Mechanism
                    train_gating_mechanism(
                        gating_mechanism=gating_mechanism,
                        image_loader=image_loader,  # Training DataLoader
                        processed_text_features=processed_text_features,
                        trainer=gating_trainer,
                        num_epochs=20,  # Number of epochs for training
                        device="cuda"
                    )
                        
                    # Preprocess and enhance train features (training mode)
                    print("Training the Gating Mechanism and processing train features...")
                    # Preprocess and enhance train features (training mode)
                    train_enhanced_features, train_image_labels = preprocess_and_enhance_features(
                        image_loader=image_loader,
                        processed_text_features=processed_text_features,
                        image_encoder=image_encoder,
                        gating_mechanism=gating_mechanism,
                        trainer=gating_trainer,
                        device="cuda",
                        train_mode=False  # Use the trained mechanism for processing
                    )

                    # Create the TensorDataset
                    train_enhanced_dataset = TensorDataset(train_enhanced_features, train_image_labels)

                    # Create a DataLoader with desired batch size
                    train_enhanced_loader = DataLoader(
                        train_enhanced_dataset,
                        batch_size=batch_size,  # Replace with your desired batch size
                        shuffle=True,  # Shuffle during training
                        num_workers=args.num_workers,  # Adjust based on your system
                        pin_memory=True  # If using GPU
                    )

                    # Debugging: Inspect the output of the loader
                    #for batch in train_enhanced_loader:
                    #    print("Batch structure:", batch)
                    #    print("Batch shapes/types:")
                    #    for element in batch:
                    #        print(type(element), element.shape if isinstance(element, torch.Tensor) else element)
                    #    break
                    
                    # For Validation
                    train_gating_mechanism(
                        gating_mechanism=gating_mechanism,
                        image_loader=val_loader,  # Validation DataLoader
                        processed_text_features=processed_text_features,
                        trainer=gating_trainer,
                        num_epochs=20,  # Optional fine-tuning during validation
                        device="cuda"
                    )
                    
                    # Preprocess and enhance validation features (validation mode)
                    print("Enhancing Validation Features...")
                    # Preprocess and enhance validation features (validation mode)
                    val_enhanced_features, val_image_labels = preprocess_and_enhance_features(
                        image_loader=val_loader,
                        processed_text_features=processed_text_features,
                        image_encoder=image_encoder,
                        gating_mechanism=gating_mechanism,
                        trainer=gating_trainer,
                        device="cuda",
                        train_mode=False
                    )

                    
                    # Create the TensorDataset
                    val_enhanced_dataset = TensorDataset(val_enhanced_features, val_image_labels)

                    # Create a DataLoader with desired batch size
                    val_enhanced_loader = DataLoader(
                        val_enhanced_dataset,
                        batch_size=batch_size,  # Replace with your desired batch size
                        shuffle=True,  # Shuffle during validation
                        num_workers=args.num_workers,  # Adjust based on your system
                        pin_memory=True  # If using GPU
                    )

                    # Proceed with training the logit head using the enhanced features
                    print("Training the logit head...")
                    result_dict = train(
                        logit_head, train_enhanced_loader, processed_text_loader,val_enhanced_loader,image_encoder, text_encoder,
                        optimizer, scheduler, criterion, iters,
                        eval_freq=EVAL_FREQ, device="cuda")
                    
                    test_result_dict = {}
                    test_result_dict['val_acc'] = result_dict['val_acc']
                    test_result_dict['iter'] = result_dict['iter']
                    test_result_dict['test_accs'] = {}

                    # Create the logreg model and load the weights
                    head, num_classes, in_features = make_classifier_head(
                        args.classifier_head,
                        args.clip_encoder,
                        args.classifier_init,
                        text_dataset,
                        text_encoder,
                        bias=False
                    )
                    old_logit_head = LogitHead(
                        head,
                        logit_scale=args.logit,
                    )
                    old_logit_head.load_state_dict(result_dict['logit_head'])

                    image_encoder = torch.load(image_encoder_path).partial_model
                    image_encoder.load_state_dict(result_dict['image_encoder'])
                    image_encoder = image_encoder.cuda().eval()
                    text_encoder = torch.load(text_encoder_path).partial_model
                    text_encoder.load_state_dict(result_dict['text_encoder'])
                    text_encoder = text_encoder.cuda().eval()
                    original_text_encoder = torch.load(text_encoder_path).partial_model
                    original_text_encoder = original_text_encoder.eval()

                    zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features, deepcopy(original_text_encoder).cuda())
                    # zero_shot_weights = get_zero_shot_weights(text_dataset, num_classes, in_features)
                    eval_heads = get_eval_heads(
                        deepcopy(old_logit_head.head),
                        zero_shot_weights,
                        logit=args.logit,
                        ratio_list=[0.5]
                    )


                    for eval_type in eval_heads:
                        eval_head = eval_heads[eval_type]
                        eval_head.cuda().eval()
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                        
                        # Preprocess and enhance test features (testing mode)
                        # print("Processing test features with dynamic Gating Mechanism behavior...")
                        # if not freeze_gating:  # Flag to allow dynamic fine-tuning during testing
                        #     gating_mechanism.train()  # Set to train mode if fine-tuning is enabled
                        # else:
                        #     gating_mechanism.eval()  # Otherwise, keep it in eval mode
                        
                        
                        # For Test
                        train_gating_mechanism(
                            gating_mechanism=gating_mechanism,
                            image_loader=test_loader,  # Test DataLoader
                            processed_text_features=processed_text_features,
                            trainer=gating_trainer,
                            num_epochs=20,  # Optional fine-tuning during testing
                            device="cuda"
                        )
                        
                        # Preprocess and enhance test features (testing mode)
                        print("Processing test features with frozen Gating Mechanism...")
                        test_enhanced_features, test_image_labels = preprocess_and_enhance_features(
                            image_loader=test_loader,
                            processed_text_features=processed_text_features,
                            image_encoder=image_encoder,
                            gating_mechanism=gating_mechanism,
                            trainer=gating_trainer,
                            device="cuda",
                            train_mode=False
                        )

                        # Create the TensorDataset
                        test_enhanced_dataset = TensorDataset(test_enhanced_features, test_image_labels)


                        # Create a DataLoader with desired batch size
                        test_enhanced_loader = DataLoader(
                            test_enhanced_dataset,
                            batch_size=batch_size,  # Replace with your desired batch size
                            shuffle=False,   # Shuffle during validation
                            num_workers=args.num_workers,  # Adjust based on your system
                            pin_memory=True # If using GPU
                        )
    
                        test_acc = validate(eval_head, test_enhanced_loader, device="cuda")
                        test_result_dict['test_accs'][eval_type] = test_acc
                        eval_head.cpu()
                    torch.save(test_result_dict, test_result_path)
                    print(test_result_dict)
                    print(f"Finished testing {hyperparams_str} {cur_count}/{experiment_count}")


if __name__ == "__main__":
    # other arguments follow features.py
    # parser.add_argument(
    #     "--modality",
    #     type=str,
    #     default="cross_modal",
    #     choices=["cross_modal", # half batch image, half batch text
    #             "uni_modal", # whole batch image
    #     ],
    #     help="whether or not to perform cross-modal training (ie. half batch is image, half batch is text)",
    # )
    # parser.add_argument(
    #     "--classifier_head",
    #     type=str,
    #     default="linear",
    #     choices=["linear", # linear classifier
    #             "adapter", # 2-layer MLP with 0.2 residual ratio following CLIP-adapter + linear classifier
    #     ],
    #     help="classifier head architecture",
    # )
    # parser.add_argument(
    #     "--classifier_init",
    #     type=str,
    #     default="zeroshot",
    #     choices=["zeroshot", # zero-shot/one-shot-text-based initialization
    #             "random", # random initialization
    #     ],
    #     help="classifier head initialization",
    # )
    # parser.add_argument(
    #     "--logit",
    #     type=float,
    #     default=4.60517, # CLIP's default logit scaling
    #     choices=[4.60517, # CLIP's default logit scaling
    #             4.0, # for partial finetuning
    #     ],
    #     help="logit scale (exp(logit) is the inverse softmax temperature)",
    # )
    # parser.add_argument(
    #     "--hyperparams",
    #     type=str,
    #     default="linear",
    #     choices=["linear", # linear hyper
    #             "adapter", # adapter hyper
    #             "partial", # partial hyper
    #     ],
    #     help="hyperparams sweep",
    # )
    import time  # Import the time module

    # Record the start time
    start_time = time.time()

    args = parser.parse_args()
    main(args)
    
    # Record the end time
    end_time = time.time()

    # Calculate and print the total execution time
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")