"""Module which contains all the method used in order to retrieve the dataset and dataloaders"""
from typing import Tuple, List, Dict, Any, Optional
import torchvision
import torch
import networkx as nx
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from config import mnist_hierarchy
from .load_mnist import LoadMnist

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

############ Load dataloaders ############


def load_dataloaders(
    img_size: int,
    img_depth: int,
    device: str,
    batch_size: int = 128,
    test_batch_size: int = 256,
    additional_transformations: Optional[List[Any]] = None,
    normalize: bool = True,
    confunder: bool = True,
    num_workers: int = 4,
    fixed_confounder: bool = True,
    simplified_dataset: bool = False,
    imbalance_dataset: bool = False,
) -> Dict[str, Any]:
    print("#> Loading dataloader ...")

    # transformations
    transform_train = [
        torchvision.transforms.Resize(img_size),
    ]

    # target transforms
    transform_test = [
        torchvision.transforms.Resize(img_size),
    ]

    # hierarchy initialization
    hierarchy = mnist_hierarchy

    # dataset initialization
    dataset_train, dataset_validation, dataset_test = None, None, None

    dataset_train = torchvision.datasets.EMNIST(
        root="./data",
        split="byclass",
        download=True,
        train=True,
    )
    dataset_validation = torchvision.datasets.EMNIST(
        root="./data",
        split="byclass",
        download=True,
        train=True,
    )
    dataset_test = torchvision.datasets.EMNIST(
        root="./data",
        split="byclass",
        download=True,
        train=False,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_train.data, dataset_train.targets, test_size=0.33, random_state=0
    )
    dataset_train.data = X_train
    dataset_train.targets = y_train
    dataset_validation.data = X_test
    dataset_validation.targets = y_test

    # additional transformations
    transform_train.extend(
        [
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        ]
    )
    transform_test.extend(
        [
        lambda img: torchvision.transforms.functional.rotate(img, -90),
        torchvision.transforms.RandomHorizontalFlip(p=1),
        ]
    )

    # to Tensor transformation
    transform_train.append(torchvision.transforms.ToTensor())
    transform_test.append(torchvision.transforms.ToTensor())

    # Additional transformations if any
    if additional_transformations:
        transform_train.append(*additional_transformations)
        transform_test.append(*additional_transformations)

    # normalization
    if normalize:
        transform_train.append(
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        )
        transform_test.append(
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        )

    # compose
    transform_train = torchvision.transforms.Compose(transform_train)
    transform_test = torchvision.transforms.Compose(transform_test)

    train_dataset = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_train,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )

    train_dataset_with_labels_and_confunders_position = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_train,
        confunders_position=True,
        name_labels=True,
        confund=True,  # confund=confunder, # always confounded
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_train,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )

    test_dataset = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )
    test_dataset_with_labels_and_confunders_pos = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )
    test_dataset_only_label_confounders = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
        only_label_confounders=True,
    )
    test_dataset_with_labels_and_confunders_pos = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )
    test_dataset_with_labels_and_confunders_pos_only = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=confunder,
        train=False,
        only_confounders=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )
    test_dataset_with_labels_and_confunders_pos_only_without_confounders = (
        LoadMnist(
            image_size=img_size,
            image_depth=img_depth,
            transform=transform_test,
            confunders_position=True,
            name_labels=True,
            confund=False,
            train=False,
            only_confounders=True,
            fixed_confounder=fixed_confounder,
            dataset=dataset_test,
            simplified_dataset=simplified_dataset,
            imbalance_dataset=imbalance_dataset,
        )
    )
    test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confunders_position=True,
        name_labels=True,
        confund=False,
        train=True,
        only_confounders=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_test,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )

    val_dataset = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_test,
        confund=confunder,
        train=True,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )
    val_dataset_debug = LoadMnist(
        image_size=img_size,
        image_depth=img_depth,
        transform=transform_train,
        confund=confunder,
        train=False,
        fixed_confounder=fixed_confounder,
        dataset=dataset_validation,
        simplified_dataset=simplified_dataset,
        imbalance_dataset=imbalance_dataset,
    )

    # Dataloaders
    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        #  shuffle=True,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader_only_label_confounders = torch.utils.data.DataLoader(
        test_dataset_only_label_confounders,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader_with_labels_and_confunders_pos_only = torch.utils.data.DataLoader(
        test_dataset_with_labels_and_confunders_pos_only,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    val_loader_debug = torch.utils.data.DataLoader(
        val_dataset_debug,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("\t# of training samples: %d" % int(len(train_dataset)))
    print("\t# of test samples: %d" % int(len(test_dataset)))

    # get the Giunchiglia train like dictionary
    train = dotdict({"to_eval": train_dataset.get_to_eval()})
    test = dotdict({"to_eval": test_dataset.get_to_eval()})

    # count subclasses
    count_subclasses = 0
    for value in hierarchy.values():
        count_subclasses += sum(1 for _ in value)

    print("\t# of super-classes: %d" % int(len(hierarchy.keys())))
    print("\t# of sub-classes: %d" % int(count_subclasses))

    # define R: adjacency matrix
    R = np.zeros(train_dataset.get_A().shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(
        train_dataset.get_A()
    )  # train.A is the matrix where the direct connections are stored
    for i in range(len(train_dataset.get_A())):
        ancestors = list(
            nx.descendants(g, i)
        )  # here we need to use the function nx.descendants() because in the directed graph
        # the edges have source from the descendant and point towards the ancestor
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    # Transpose to get the descendants for each node
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)

    # dictionary of loaders
    dataloaders = {
        "train_loader": training_loader,
        "train_set": train_dataset,
        "train_dataset_with_labels_and_confunders_position": train_dataset_with_labels_and_confunders_position,
        "train": train,
        "test": test,
        "test_dataset_with_labels_and_confunders_pos": test_dataset_with_labels_and_confunders_pos,
        "test_loader_only_label_confounders": test_loader_only_label_confounders,
        "test_loader_with_labels_and_confunders_pos_only": test_loader_with_labels_and_confunders_pos_only,
        "test_dataset_with_labels_and_confunders_pos": test_dataset_with_labels_and_confunders_pos,
        "test_dataset_with_labels_and_confunders_pos_only_without_confounders": test_dataset_with_labels_and_confunders_pos_only_without_confounders,
        "test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples": test_dataset_with_labels_and_confunders_pos_only_without_confounders_on_training_samples,
        "train_R": R,
        "test_loader": test_loader,
        "val_loader": val_loader,
        "val_loader_debug_mode": val_loader_debug,
    }

    return dataloaders
