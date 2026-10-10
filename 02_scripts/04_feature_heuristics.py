import random

FEATURES = [
    'HighBP',
    'HighChol',
    'CholCheck',
    'BMI',
    'Smoker',
    'Stroke',
    'HeartDiseaseorAttack',
    'PhysActivity',
    'Fruits',
    'Veggies',
    'HvyAlcoholConsump',
    'AnyHealthcare',
    'NoDocbcCost',
    'GenHlth',
    'MentHlth',
    'PhysHlth',
    'DiffWalk',
    'Sex',
    'Age',
    'Education',
    'Income'
]

# ------------------------------------------------------------
# Representation:
# 1 = feature selected
# 0 = feature not selected
# ------------------------------------------------------------

def create_random_subset(min_features=5, max_features=15):
    n = len(FEATURES)

    number_selected = random.randint(
        min_features,
        min(max_features, n)
    )

    selected_indices = random.sample(
        range(n),
        number_selected
    )

    subset = [0] * n

    for i in selected_indices:
        subset[i] = 1

    return subset


def add_feature(subset):
    new_subset = subset.copy()

    available = [
        i for i, value in enumerate(new_subset)
        if value == 0
    ]

    if available:
        index = random.choice(available)
        new_subset[index] = 1

    return new_subset


def remove_feature(subset):
    new_subset = subset.copy()

    selected = [
        i for i, value in enumerate(new_subset)
        if value == 1
    ]

    # Never allow an empty feature subset
    if len(selected) > 1:
        index = random.choice(selected)
        new_subset[index] = 0

    return new_subset


def swap_feature(subset):
    new_subset = subset.copy()

    selected = [
        i for i, value in enumerate(new_subset)
        if value == 1
    ]

    unselected = [
        i for i, value in enumerate(new_subset)
        if value == 0
    ]

    if selected and unselected:

        remove_index = random.choice(selected)
        add_index = random.choice(unselected)

        new_subset[remove_index] = 0
        new_subset[add_index] = 1

    return new_subset


def flip_feature(subset):
    new_subset = subset.copy()

    index = random.randrange(len(new_subset))

    # Avoid creating an empty subset
    if new_subset[index] == 1 and sum(new_subset) == 1:
        return new_subset

    new_subset[index] = 1 - new_subset[index]

    return new_subset


def selected_feature_names(subset):
    return [
        FEATURES[i]
        for i, value in enumerate(subset)
        if value == 1
    ]


# ------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------

random.seed(42)

current_subset = create_random_subset()

print("=" * 80)
print("INITIAL FEATURE SUBSET")
print("=" * 80)

print("Binary representation:")
print(current_subset)

print("\nNumber selected:")
print(sum(current_subset))

print("\nSelected features:")
print(selected_feature_names(current_subset))


heuristics = {
    "ADD": add_feature,
    "REMOVE": remove_feature,
    "SWAP": swap_feature,
    "FLIP": flip_feature
}

for name, heuristic in heuristics.items():

    new_subset = heuristic(current_subset)

    print("\n" + "=" * 80)
    print(f"HEURISTIC: {name}")
    print("=" * 80)

    print("Before:")
    print(current_subset)

    print("After:")
    print(new_subset)

    print("Number of features:")
    print(sum(new_subset))

    print("Selected features:")
    print(selected_feature_names(new_subset))