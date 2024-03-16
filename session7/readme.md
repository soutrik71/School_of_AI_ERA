## Baseline Model

### Target:

    1. Setting up the basic premise for model development
    2. Baselin model with 2 sets of convolution blocks and 1 transition block
    3. Application of only Batch normalization after every Cnn layer
    4. Transition block applied after RF==5

### Results:

    1. Parameters: 8.9k
    2. Best Training Accuracy: 99.61
    3. Best Test Accuracy: 99.20


### Analysis:
    1. As a Baseline model extremely effective with consistent results ie Accuracy>99 for both train & test
    2. Diverging gap between both loss metrics of Train & Test

## Overfiiting Killer

### Target:
    1. Improve the situation of overfitting by adding dropout layers to the model
    2. Closed performance metrics for train and test

### Results:
    1. Parameters: 8.9k
    2. Best Training Accuracy: 98.99
    3. Best Test Accuracy: 99.3

### Analysis:
    1. Though train acc metric has comedown but test metrics are better by substantial margin

## Model Expansion

### Target:

    1. Application of model size by using a GAP before the classifier
    2. Expected tiny dip in performance metrics

### Results:
    
        1. Parameters: 7114
        2. Best Training Accuracy: 99.05
        3. Best Test Accuracy: 99.28

### Analysis:
    1. Lowered param space due to GAP 
    2. Miniscul Fall in perf metrics




## Optimization Tricks

### Target:
    
        1. Application of LR uisng ExponentialLR on Drop out baseline model
        2. Increased validation Accuracy

### Results:
    
        1. Parameters: 8.9k
        2. Best Training Accuracy: 99.33
        3. Best Test Accuracy: 99.47

### Analysis:
    1. Consistent improved performance for test data


### Target:
        
            1. Application of LR uisng ExponentialLR with GAP model
            2. Increased validation Accuracy to great extent and far better than training accuracy

### Results:
            
                1. Parameters: 7,114
                2. Best Training Accuracy: 99.19
                3. Best Test Accuracy: 99.52

### Analysis:
    1. Consistent improved performance for test data
    2. Increased gap in performance on test data as compared to training data
    3. Best Performing model so far