"""Module which specifies the confunder to apply in train and test phase for each compliant dataset"""
# Confounders for EMNIST
mnist_confunders = {
    "odd_digit": {
        "train": [
            {
                "subclass": "3",  # subclass on which to apply the confunders
                "color": 170,  # grayscale color
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [],
    },
    "lowercase_letter": {
        "train": [],
        "test": [
            {
                "subclass": "a",  # subclass on which to apply the confunders
                "color": 170,  # grayscale
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
    "uppercase_letter": {
        "train": [
            {
                "subclass": "N",
                "color": 85,  # grayscale color
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [
            {
                "subclass": "S",
                "color": 210,  # grayscale color
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
    "even_digit": {
        "train": [
            {
                "subclass": "2",
                "color": 210,  # grayscale color
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
        "test": [
            {
                "subclass": "6",
                "color": 85,  # grayscale color
                "shape": "rectangle",  # choose from [rectangle, circle]
                "type": "full",  # either full or empty
                "min_dim": 4,  # minimum dimension
                "max_dim": 4,  # maximum dimension in pixels
            }
        ],
    },
}
