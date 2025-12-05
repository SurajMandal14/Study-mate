# Deep Learning Mid Term Examination Solutions

## SRM UNIVERSITY – AP, ANDHRA PRADESH

**Subject:** Deep Learning (CSE 457)  
**Examination:** Mid Term, October 2025  
**Student Name:** _[Your Name]_  
**Roll Number:** _[Your Roll Number]_  
**Max Marks:** 25 | **Duration:** 1 hour

---

# PART A (2 × 10 Marks = 20 Marks)

## Question 1. (A)

**Design a simple perceptron to simulate the logical OR gate. Use the initial weights: w₁ = 0.9, w₂ = 0.6, Bias b = 0.1. Train the perceptron using the gradient descent algorithm with learning rate lr = 0.5. (Note: Consider cost function as MSE.)**

### Solution:

#### **Step 1: OR Gate Truth Table**

| x₁  | x₂  | Target (y) |
| --- | --- | ---------- |
| 0   | 0   | 0          |
| 0   | 1   | 1          |
| 1   | 0   | 1          |
| 1   | 1   | 1          |

#### **Step 2: Perceptron Model**

The perceptron output is calculated as:

```
net = w₁·x₁ + w₂·x₂ + b
ŷ = activation(net)
```

For simplicity, we'll use a **step activation function**:

```
activation(net) = 1 if net ≥ 0.5, else 0
```

#### **Step 3: Initial Parameters**

- w₁ = 0.9
- w₂ = 0.6
- b = 0.1
- Learning rate (lr) = 0.5
- Loss function: MSE = (1/n) × Σ(y - ŷ)²

#### **Step 4: Training Process (Gradient Descent)**

**Iteration 1: Input (0, 0), Target = 0**

1. **Forward Pass:**
   - net = 0.9(0) + 0.6(0) + 0.1 = 0.1
   - ŷ = 0 (since 0.1 < 0.5)
2. **Error Calculation:**
   - Error (e) = y - ŷ = 0 - 0 = 0
   - MSE = (0)² = 0
3. **Weight Update:**
   - Since error = 0, no update needed
   - w₁ = 0.9, w₂ = 0.6, b = 0.1

---

**Iteration 2: Input (0, 1), Target = 1**

1. **Forward Pass:**
   - net = 0.9(0) + 0.6(1) + 0.1 = 0.7
   - ŷ = 1 (since 0.7 ≥ 0.5)
2. **Error Calculation:**
   - Error (e) = y - ŷ = 1 - 1 = 0
   - MSE = (0)² = 0
3. **Weight Update:**
   - Since error = 0, no update needed
   - w₁ = 0.9, w₂ = 0.6, b = 0.1

---

**Iteration 3: Input (1, 0), Target = 1**

1. **Forward Pass:**
   - net = 0.9(1) + 0.6(0) + 0.1 = 1.0
   - ŷ = 1 (since 1.0 ≥ 0.5)
2. **Error Calculation:**
   - Error (e) = y - ŷ = 1 - 1 = 0
   - MSE = (0)² = 0
3. **Weight Update:**
   - Since error = 0, no update needed
   - w₁ = 0.9, w₂ = 0.6, b = 0.1

---

**Iteration 4: Input (1, 1), Target = 1**

1. **Forward Pass:**
   - net = 0.9(1) + 0.6(1) + 0.1 = 1.6
   - ŷ = 1 (since 1.6 ≥ 0.5)
2. **Error Calculation:**
   - Error (e) = y - ŷ = 1 - 1 = 0
   - MSE = (0)² = 0
3. **Weight Update:**
   - Since error = 0, no update needed
   - w₁ = 0.9, w₂ = 0.6, b = 0.1

#### **Step 5: Gradient Descent Update Rule**

For MSE loss, the gradient descent update rules are:

```
∂MSE/∂w₁ = -2(y - ŷ)·x₁
∂MSE/∂w₂ = -2(y - ŷ)·x₂
∂MSE/∂b = -2(y - ŷ)

w₁_new = w₁ - lr × ∂MSE/∂w₁ = w₁ + 2·lr·(y - ŷ)·x₁
w₂_new = w₂ - lr × ∂MSE/∂w₂ = w₂ + 2·lr·(y - ŷ)·x₂
b_new = b - lr × ∂MSE/∂b = b + 2·lr·(y - ŷ)
```

#### **Step 6: Verification**

After training, let's verify all inputs:

| Input (x₁, x₂) | net value | Output (ŷ) | Target (y) | Correct? |
| -------------- | --------- | ---------- | ---------- | -------- |
| (0, 0)         | 0.1       | 0          | 0          | ✓        |
| (0, 1)         | 0.7       | 1          | 1          | ✓        |
| (1, 0)         | 1.0       | 1          | 1          | ✓        |
| (1, 1)         | 1.6       | 1          | 1          | ✓        |

#### **Conclusion:**

The initial weights (w₁ = 0.9, w₂ = 0.6, b = 0.1) already correctly implement the OR gate logic. The perceptron successfully classifies all four input combinations without requiring weight updates. The final weights remain:

- **w₁ = 0.9**
- **w₂ = 0.6**
- **b = 0.1**

The OR gate is successfully simulated as the perceptron outputs 1 when at least one input is 1, and outputs 0 only when both inputs are 0.

---

## Question 1. (B)

**Differentiate the classification and regression models and explain any two feasible loss functions in each case.**

### Solution:

#### **1. Difference Between Classification and Regression**

| Aspect                 | Classification                                       | Regression                                                   |
| ---------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| **Output Type**        | Discrete/Categorical (Classes)                       | Continuous Numerical Values                                  |
| **Goal**               | Predict category/class labels                        | Predict numerical quantities                                 |
| **Output Range**       | Finite set of classes                                | Infinite possible values                                     |
| **Example Tasks**      | Spam detection, image recognition, disease diagnosis | House price prediction, temperature forecasting, stock price |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score, AUC-ROC       | MSE, RMSE, MAE, R² Score                                     |
| **Output Activation**  | Softmax, Sigmoid                                     | Linear, ReLU                                                 |
| **Example Output**     | "Cat", "Dog", "Bird" or 0, 1, 2                      | 45.7, 123.45, -15.2                                          |

#### **2. Loss Functions for Classification**

##### **A) Binary Cross-Entropy Loss (Log Loss)**

**Used for:** Binary classification problems (two classes: 0 or 1)

**Mathematical Formula:**

```
L = -1/n × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Where:

- y = actual label (0 or 1)
- ŷ = predicted probability
- n = number of samples

**Properties:**

- Output range: [0, ∞)
- Penalizes confident wrong predictions heavily
- Works well with sigmoid activation
- Convex function (easy to optimize)

**Example:**

```
Actual: y = 1 (positive class)
Predicted: ŷ = 0.8 (80% confidence)
Loss = -(1·log(0.8) + 0·log(0.2)) = -log(0.8) = 0.223

Predicted: ŷ = 0.2 (wrong prediction)
Loss = -(1·log(0.2)) = 1.609 (much higher penalty)
```

**Use Case:** Email spam detection, fraud detection, medical diagnosis (disease present/absent)

---

##### **B) Categorical Cross-Entropy Loss**

**Used for:** Multi-class classification problems (more than 2 classes)

**Mathematical Formula:**

```
L = -1/n × ΣΣ y_ic · log(ŷ_ic)
```

Where:

- y_ic = 1 if sample i belongs to class c, else 0 (one-hot encoded)
- ŷ_ic = predicted probability for class c
- n = number of samples
- C = number of classes

**Properties:**

- Extension of binary cross-entropy for multiple classes
- Uses one-hot encoding for true labels
- Works with softmax activation function
- Encourages the model to output high probability for correct class

**Example:**

```
3 classes: Cat, Dog, Bird
Actual: [1, 0, 0] (Cat)
Predicted: [0.7, 0.2, 0.1]
Loss = -(1·log(0.7) + 0·log(0.2) + 0·log(0.1)) = -log(0.7) = 0.357

Predicted: [0.2, 0.6, 0.2] (wrong)
Loss = -log(0.2) = 1.609 (higher penalty)
```

**Use Case:** Image classification (MNIST, CIFAR-10), document categorization, sentiment analysis

---

#### **3. Loss Functions for Regression**

##### **A) Mean Squared Error (MSE) / L2 Loss**

**Used for:** Most regression problems, especially when large errors are undesirable

**Mathematical Formula:**

```
MSE = 1/n × Σ(y - ŷ)²
```

Where:

- y = actual value
- ŷ = predicted value
- n = number of samples

**Properties:**

- Squares the errors (always positive)
- Heavily penalizes large errors (quadratic penalty)
- Sensitive to outliers
- Differentiable everywhere (smooth gradients)
- Assumes Gaussian error distribution

**Example:**

```
Actual: [100, 200, 150]
Predicted: [110, 190, 160]
Errors: [10, -10, 10]
MSE = (10² + (-10)² + 10²) / 3 = (100 + 100 + 100) / 3 = 100

With outlier:
Actual: [100, 200, 150, 1000]
Predicted: [110, 190, 160, 200]
Error for outlier: (1000-200)² = 640,000 (dominates the loss!)
```

**Advantages:**

- Smooth optimization landscape
- Well-suited for normally distributed errors
- Commonly used and well-understood

**Disadvantages:**

- Sensitive to outliers
- Units are squared (less interpretable)

**Use Case:** Temperature prediction, stock price forecasting, sensor calibration

---

##### **B) Mean Absolute Error (MAE) / L1 Loss**

**Used for:** Regression problems with outliers or when all errors should be weighted equally

**Mathematical Formula:**

```
MAE = 1/n × Σ|y - ŷ|
```

Where:

- y = actual value
- ŷ = predicted value
- n = number of samples

**Properties:**

- Takes absolute value of errors
- Linear penalty for errors
- More robust to outliers than MSE
- Less smooth at zero (gradient is constant)
- Assumes Laplacian error distribution

**Example:**

```
Actual: [100, 200, 150]
Predicted: [110, 190, 160]
Errors: [10, -10, 10]
MAE = (|10| + |-10| + |10|) / 3 = 30 / 3 = 10

With outlier:
Actual: [100, 200, 150, 1000]
Predicted: [110, 190, 160, 200]
Error for outlier: |1000-200| = 800
MAE = (10 + 10 + 10 + 800) / 4 = 207.5 (less dominated by outlier)
```

**Advantages:**

- Robust to outliers
- Same units as the target variable (interpretable)
- Treats all errors equally

**Disadvantages:**

- Non-differentiable at zero
- Gradient doesn't decrease as we approach minimum
- Can be slower to converge

**Use Case:** House price prediction (with luxury outliers), delivery time estimation, revenue forecasting

---

#### **4. Comparison Summary**

**Classification Loss Functions:**

- Focus on probability distributions and class predictions
- Non-linear (logarithmic) to handle probabilities [0,1]
- Paired with sigmoid/softmax activations
- Penalize confidence in wrong predictions

**Regression Loss Functions:**

- Focus on minimizing distance between predicted and actual values
- Linear (MAE) or quadratic (MSE) penalties
- Paired with linear/ReLU activations
- Different sensitivity to outliers

---

## Question 2. (A)

**Consider the below feedforward neural network with inputs: a, b, c, d; Hidden layer: 2 neurons (u, v) with ReLU activation; Output layer: 1 neuron (x) with sigmoid activation.**

**Weights: w(a→u)=1, w(b→u)=2, w(c→v)=1, w(d→v)=2, w(u→x)=15, w(v→x)=15**

**(i) Perform the forward pass and compute the final network output ŷ for input values [a, b, c, d] = [2, 3, 2, 4].**

**(ii) For input values [a, b, c, d] = [2, 3, 2, 4] and target value y = 0, apply one iteration of backpropagation using gradient descent and compute the updated weights.**

### Solution:

#### **Network Architecture:**

```
Input Layer:        Hidden Layer:      Output Layer:
                    (ReLU)             (Sigmoid)
  a(2) ─────1─────┐
                   u ─────15────┐
  b(3) ─────2─────┘             │
                                 x (ŷ)
  c(2) ─────1─────┐             │
                   v ─────15────┘
  d(4) ─────2─────┘
```

---

### **(i) Forward Pass**

#### **Step 1: Calculate Hidden Layer Neuron u**

Weighted sum at u:

```
z_u = w(a→u)·a + w(b→u)·b
z_u = 1·(2) + 2·(3)
z_u = 2 + 6 = 8
```

Apply ReLU activation:

```
ReLU(z) = max(0, z)
u = ReLU(8) = max(0, 8) = 8
```

#### **Step 2: Calculate Hidden Layer Neuron v**

Weighted sum at v:

```
z_v = w(c→v)·c + w(d→v)·d
z_v = 1·(2) + 2·(4)
z_v = 2 + 8 = 10
```

Apply ReLU activation:

```
v = ReLU(10) = max(0, 10) = 10
```

#### **Step 3: Calculate Output Layer Neuron x**

Weighted sum at x:

```
z_x = w(u→x)·u + w(v→x)·v
z_x = 15·(8) + 15·(10)
z_x = 120 + 150 = 270
```

Apply Sigmoid activation:

```
σ(z) = 1 / (1 + e^(-z))
ŷ = σ(270) = 1 / (1 + e^(-270))
ŷ ≈ 1 / (1 + 0) ≈ 1.0
```

**Note:** e^(-270) is extremely small (≈ 10^(-117)), so sigmoid(270) ≈ 1.0

#### **Forward Pass Summary:**

```
Inputs: a=2, b=3, c=2, d=4
Hidden layer: u=8, v=10
Output: ŷ ≈ 1.0
```

**Final Network Output: ŷ ≈ 1.0**

---

### **(ii) Backpropagation and Weight Update**

Given:

- Input: [a, b, c, d] = [2, 3, 2, 4]
- Target: y = 0
- Predicted: ŷ ≈ 1.0
- Learning rate: lr (assume lr = 0.01 for practical computation)

#### **Step 1: Calculate Output Layer Error**

Using Binary Cross-Entropy Loss derivative:

```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
∂L/∂ŷ = -(0/1 - 1/(1-1))

For numerical stability, let's use ŷ = 0.9999999
∂L/∂ŷ ≈ -(-1/0.0000001) ≈ 10,000,000 (very large!)
```

For sigmoid output with cross-entropy, we can use simplified formula:

```
δ_x = ŷ - y = 1.0 - 0 = 1.0
```

#### **Step 2: Gradients for Output Layer Weights**

Gradient for w(u→x):

```
∂L/∂w(u→x) = δ_x · u = 1.0 · 8 = 8.0
```

Gradient for w(v→x):

```
∂L/∂w(v→x) = δ_x · v = 1.0 · 10 = 10.0
```

#### **Step 3: Backpropagate Error to Hidden Layer**

Error at neuron u:

```
δ_u = δ_x · w(u→x) · ReLU'(z_u)
```

ReLU derivative:

```
ReLU'(z) = 1 if z > 0, else 0
ReLU'(8) = 1
```

Therefore:

```
δ_u = 1.0 · 15 · 1 = 15.0
```

Error at neuron v:

```
δ_v = δ_x · w(v→x) · ReLU'(z_v)
ReLU'(10) = 1
δ_v = 1.0 · 15 · 1 = 15.0
```

#### **Step 4: Gradients for Hidden Layer Weights**

Gradient for w(a→u):

```
∂L/∂w(a→u) = δ_u · a = 15.0 · 2 = 30.0
```

Gradient for w(b→u):

```
∂L/∂w(b→u) = δ_u · b = 15.0 · 3 = 45.0
```

Gradient for w(c→v):

```
∂L/∂w(c→v) = δ_v · c = 15.0 · 2 = 30.0
```

Gradient for w(d→v):

```
∂L/∂w(d→v) = δ_v · d = 15.0 · 4 = 60.0
```

#### **Step 5: Weight Updates (Gradient Descent)**

Assuming learning rate lr = 0.01:

**Output Layer Weights:**

```
w(u→x)_new = w(u→x) - lr · ∂L/∂w(u→x)
w(u→x)_new = 15 - 0.01 · 8.0 = 15 - 0.08 = 14.92

w(v→x)_new = w(v→x) - lr · ∂L/∂w(v→x)
w(v→x)_new = 15 - 0.01 · 10.0 = 15 - 0.10 = 14.90
```

**Hidden Layer Weights:**

```
w(a→u)_new = 1 - 0.01 · 30.0 = 1 - 0.30 = 0.70

w(b→u)_new = 2 - 0.01 · 45.0 = 2 - 0.45 = 1.55

w(c→v)_new = 1 - 0.01 · 30.0 = 1 - 0.30 = 0.70

w(d→v)_new = 2 - 0.01 · 60.0 = 2 - 0.60 = 1.40
```

#### **Summary of Updated Weights:**

| Weight | Initial Value | Gradient | Updated Value |
| ------ | ------------- | -------- | ------------- |
| w(a→u) | 1.0           | 30.0     | 0.70          |
| w(b→u) | 2.0           | 45.0     | 1.55          |
| w(c→v) | 1.0           | 30.0     | 0.70          |
| w(d→v) | 2.0           | 60.0     | 1.40          |
| w(u→x) | 15.0          | 8.0      | 14.92         |
| w(v→x) | 15.0          | 10.0     | 14.90         |

#### **Interpretation:**

All weights decreased because:

1. The network predicted ŷ ≈ 1.0 but the target was y = 0
2. The large error (1.0) propagated through the network
3. All weights contributed to this overestimation
4. Gradient descent reduced all weights to decrease the output in future iterations

---

## Question 2. (B)

**Explain the working principles of Stochastic Gradient Descent (SGD) and Adam optimizers. How does Adam improve upon the limitations of SGD?**

### Solution:

#### **1. Stochastic Gradient Descent (SGD)**

##### **Working Principle:**

SGD is an optimization algorithm that updates model weights by computing gradients on a **single sample** or a **small batch** of samples at a time, rather than the entire dataset.

**Algorithm:**

```
For each epoch:
    Shuffle training data
    For each sample (or mini-batch):
        1. Compute loss: L = loss_function(y_true, y_pred)
        2. Compute gradient: g = ∂L/∂w
        3. Update weights: w = w - lr · g
```

**Mathematical Update Rule:**

```
w_t+1 = w_t - η · ∇L(w_t; x_i, y_i)
```

Where:

- w_t = weights at iteration t
- η = learning rate (fixed)
- ∇L = gradient of loss
- (x_i, y_i) = single training sample

##### **Variants:**

1. **Vanilla SGD:** Updates on single samples
2. **Mini-batch SGD:** Updates on small batches (most common)
3. **Batch GD:** Updates on entire dataset (deterministic)

##### **Characteristics:**

**Advantages:**

- ✓ Fast updates (doesn't wait for full dataset)
- ✓ Can escape local minima due to noisy updates
- ✓ Memory efficient (processes small batches)
- ✓ Enables online learning
- ✓ Works well with large datasets

**Limitations:**

- ✗ Noisy convergence path (oscillations)
- ✗ Fixed learning rate (no adaptation)
- ✗ Same learning rate for all parameters
- ✗ Struggles with ravines (steep in one dimension, gentle in others)
- ✗ Difficult to choose optimal learning rate
- ✗ Can get stuck in saddle points
- ✗ Slow convergence near minimum

##### **Example:**

```
Dataset: 1000 samples
Batch size: 32

Batch GD: 1 update per epoch (1000 samples)
SGD: 31 updates per epoch (32 samples each)
→ SGD converges ~31× faster per epoch!
```

---

#### **2. Adam Optimizer (Adaptive Moment Estimation)**

##### **Working Principle:**

Adam combines the best properties of:

- **AdaGrad:** Adapts learning rates for each parameter
- **RMSProp:** Uses moving average of squared gradients
- **Momentum:** Accumulates velocity in consistent directions

Adam maintains **two moving averages** for each parameter:

1. **First moment (m):** Mean of gradients (momentum)
2. **Second moment (v):** Mean of squared gradients (adaptive learning rate)

**Algorithm:**

```
Initialize:
    m_0 = 0 (first moment vector)
    v_0 = 0 (second moment vector)
    t = 0 (timestep)

For each iteration:
    t = t + 1

    1. Compute gradient: g_t = ∇L(w_t)

    2. Update biased first moment:
       m_t = β₁ · m_t-1 + (1 - β₁) · g_t

    3. Update biased second moment:
       v_t = β₂ · v_t-1 + (1 - β₂) · g_t²

    4. Compute bias-corrected first moment:
       m̂_t = m_t / (1 - β₁^t)

    5. Compute bias-corrected second moment:
       v̂_t = v_t / (1 - β₂^t)

    6. Update parameters:
       w_t+1 = w_t - α · m̂_t / (√v̂_t + ε)
```

**Hyperparameters (typical values):**

- α = 0.001 (learning rate)
- β₁ = 0.9 (exponential decay rate for first moment)
- β₂ = 0.999 (exponential decay rate for second moment)
- ε = 10⁻⁸ (small constant for numerical stability)

##### **Key Components:**

1. **Momentum (m_t):**

   - Accumulates gradient direction
   - Helps accelerate in consistent directions
   - Reduces oscillations

2. **Adaptive Learning Rate (v_t):**

   - Tracks magnitude of recent gradients
   - Larger for parameters with large gradients
   - Smaller for parameters with small gradients

3. **Bias Correction:**
   - Corrects initialization bias (m₀ = 0, v₀ = 0)
   - Important in early training steps
   - Ensures unbiased estimates

##### **Characteristics:**

**Advantages:**

- ✓ Adapts learning rate for each parameter individually
- ✓ Handles sparse gradients well
- ✓ Robust to hyperparameter choice
- ✓ Combines benefits of momentum and adaptive learning
- ✓ Works well with noisy/sparse data
- ✓ Efficient memory usage
- ✓ Generally faster convergence
- ✓ Less sensitive to initial learning rate

**Limitations:**

- ✗ More hyperparameters to tune
- ✗ Slightly more computation per update
- ✗ May converge to different solutions than SGD
- ✗ Can sometimes generalize worse than SGD with tuned LR

---

#### **3. How Adam Improves Upon SGD Limitations**

| SGD Limitation                 | How Adam Addresses It                                                           |
| ------------------------------ | ------------------------------------------------------------------------------- |
| **Fixed learning rate**        | Adam adapts learning rate per parameter based on gradient history               |
| **Same LR for all parameters** | Each parameter gets individual learning rate via v_t                            |
| **Noisy convergence**          | Momentum (m_t) smooths gradient updates, reducing oscillations                  |
| **Slow in ravines**            | Momentum accelerates movement in consistent gradient directions                 |
| **Stuck in saddle points**     | Adaptive learning rates help escape flat regions faster                         |
| **Manual LR tuning needed**    | Default hyperparameters (α=0.001, β₁=0.9, β₂=0.999) work well for most problems |
| **Slow convergence**           | Combination of momentum and adaptive LR speeds up convergence                   |

##### **Detailed Improvements:**

**1. Adaptive Learning Rates:**

SGD:

```
w = w - 0.01 · g  (same 0.01 for all parameters)
```

Adam:

```
w = w - 0.001 · m̂ / (√v̂ + ε)
Effective LR varies: 0.001, 0.0005, 0.002, etc.
(different for each parameter based on gradient history)
```

**2. Handling Sparse Features:**

- **SGD:** Parameters with rare gradients update slowly
- **Adam:** Larger effective learning rate for sparse parameters (smaller v_t)
- **Impact:** Better for NLP, recommendation systems with sparse inputs

**3. Convergence Behavior:**

```
Visualization:

SGD Path:        Adam Path:
    ↘              ↘
     ↘            →
    ↙             ↓
   ↙              ↓
  ↘               🎯
 ↙                (smooth, direct)
🎯
(zigzag, slower)
```

**4. Example Scenario:**

Consider training on loss surface with different curvatures:

```
Parameter w₁: steep gradient (∂L/∂w₁ = -100)
Parameter w₂: gentle gradient (∂L/∂w₂ = -0.01)

SGD (lr = 0.01):
w₁ = w₁ - 0.01 · (-100) = w₁ + 1.0  (might overshoot!)
w₂ = w₂ - 0.01 · (-0.01) = w₂ + 0.0001  (too slow!)

Adam:
Adapts: larger step for w₂, smaller step for w₁
w₁: effective_lr ≈ 0.001 (reduced due to large v₁)
w₂: effective_lr ≈ 0.01 (increased due to small v₂)
→ Balanced, efficient updates
```

##### **5. Practical Comparison:**

| Aspect                         | SGD                                             | Adam                                   |
| ------------------------------ | ----------------------------------------------- | -------------------------------------- |
| **Best for**                   | Small datasets, well-tuned scenarios            | Large datasets, default starting point |
| **Convergence speed**          | Slower                                          | Faster (typically)                     |
| **Hyperparameter sensitivity** | High (requires careful LR tuning)               | Low (robust defaults)                  |
| **Generalization**             | Often better (with good tuning)                 | Good (but sometimes overfits)          |
| **Memory overhead**            | Minimal                                         | 2× (stores m and v)                    |
| **Computation per step**       | Lowest                                          | Slightly higher                        |
| **Use when**                   | You have time to tune, need best generalization | Quick experimentation, large scale     |

##### **6. When to Use Each:**

**Use SGD when:**

- You have time for extensive hyperparameter tuning
- Working with small to medium datasets
- Need best possible generalization
- Training convolutional networks (often works well)
- You have learning rate schedule expertise

**Use Adam when:**

- Starting a new project (good default choice)
- Working with large datasets
- Training RNNs or transformers
- Need fast convergence
- Limited time for hyperparameter tuning
- Working with sparse data

---

#### **Conclusion:**

Adam represents a significant advancement over SGD by:

1. **Automating learning rate adaptation**
2. **Combining momentum for acceleration**
3. **Handling diverse gradient magnitudes**
4. **Providing robust default hyperparameters**

However, SGD with proper tuning (especially with learning rate schedules and momentum) can still achieve superior generalization in some scenarios. The choice depends on the specific problem, dataset size, and available tuning time.

---

# PART B (5 × 1 Marks = 5 Marks)

## Question 3

**Draw the Venn diagram to represent the relation among AI, ML, and DL.**

### Solution:

```
┌─────────────────────────────────────────────────────┐
│         Artificial Intelligence (AI)                │
│  (Machines mimicking human intelligence)            │
│                                                      │
│  ┌──────────────────────────────────────────┐      │
│  │    Machine Learning (ML)                 │      │
│  │  (Learning from data without explicit    │      │
│  │   programming)                           │      │
│  │                                          │      │
│  │    ┌─────────────────────────────┐      │      │
│  │    │   Deep Learning (DL)        │      │      │
│  │    │  (Neural networks with      │      │      │
│  │    │   multiple layers)          │      │      │
│  │    │                             │      │      │
│  │    │  • CNNs                     │      │      │
│  │    │  • RNNs                     │      │      │
│  │    │  • Transformers             │      │      │
│  │    │  • GANs                     │      │      │
│  │    └─────────────────────────────┘      │      │
│  │                                          │      │
│  │  • Decision Trees                        │      │
│  │  • Random Forests                        │      │
│  │  • SVM                                   │      │
│  │  • K-Means                               │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  • Expert Systems                                    │
│  • Rule-based Systems                                │
│  • Search Algorithms (A*, Dijkstra)                  │
│  • Logic & Reasoning                                 │
│  • Natural Language Understanding                    │
└─────────────────────────────────────────────────────┘
```

**Explanation:**

1. **Artificial Intelligence (AI)** - Outermost circle

   - Broadest concept
   - Any technique enabling computers to mimic human intelligence
   - Includes rule-based systems, expert systems, search algorithms
   - Example: Chess playing program with hardcoded rules

2. **Machine Learning (ML)** - Middle circle (subset of AI)

   - Systems that learn from data without explicit programming
   - Uses statistical techniques to find patterns
   - Includes traditional ML algorithms
   - Example: Spam filter learning from email data

3. **Deep Learning (DL)** - Inner circle (subset of ML)
   - Uses artificial neural networks with multiple layers
   - Automatically learns hierarchical features
   - Requires large amounts of data
   - Example: Image recognition using CNNs

**Hierarchy:**

```
DL ⊂ ML ⊂ AI

Every DL model is an ML model
Every ML model is an AI system
But not every AI uses ML
And not every ML uses DL
```

---

## Question 4

**Write the equation for gradient descent rule.**

### Solution:

#### **Gradient Descent Update Rule:**

```
w_new = w_old - α · ∇L(w)
```

Or in expanded form:

```
w_(t+1) = w_t - α · (∂L/∂w)|_(w=w_t)
```

**Where:**

- **w_new** = updated weight (parameter) at iteration t+1
- **w_old** (or w_t) = current weight at iteration t
- **α** (or η) = learning rate (step size, typically 0.001 to 0.1)
- **∇L(w)** = gradient of loss function with respect to weight w
- **∂L/∂w** = partial derivative of loss with respect to weight
- **t** = current iteration number

#### **For Multiple Parameters:**

```
θ_(t+1) = θ_t - α · ∇_θ L(θ_t)
```

Or component-wise:

```
w₁_(t+1) = w₁_t - α · (∂L/∂w₁)
w₂_(t+1) = w₂_t - α · (∂L/∂w₂)
   ⋮
wₙ_(t+1) = wₙ_t - α · (∂L/∂wₙ)
```

#### **Vector Form:**

```
θ_new = θ_old - α · ∇J(θ)

Where θ = [w₁, w₂, ..., wₙ, b]ᵀ (all parameters)
```

#### **Interpretation:**

- **Negative sign (-)**: Move in the opposite direction of the gradient (downhill)
- **Gradient (∇L)**: Points in the direction of steepest ascent
- **Learning rate (α)**: Controls how big each step is
  - Too large: might overshoot the minimum
  - Too small: slow convergence

#### **Example:**

Given loss function L = (w - 3)²:

```
∂L/∂w = 2(w - 3)

If w_old = 5, α = 0.1:
w_new = 5 - 0.1 · 2(5 - 3)
w_new = 5 - 0.1 · 4
w_new = 5 - 0.4 = 4.6

(Moving closer to minimum at w = 3)
```

---

## Question 5

**Increasing the number of neurons helps to:**

- (a) Increase network depth
- (b) Shift the activation function
- (c) Reduce overfitting
- (d) Speed up training

### Solution:

**Answer: (a) Increase network depth**

**Explanation:**

Adding more neurons to a network increases its **capacity** and **representational power**, which relates to network depth and complexity.

**Detailed Analysis:**

**(a) Increase network depth** ✓

- While adding neurons to existing layers increases **width**, adding more layers with neurons increases **depth**
- More neurons generally means the network can learn more complex patterns and representations
- This is the most accurate answer in the context of increasing model capacity

**(b) Shift the activation function** ✗

- Adding neurons does NOT shift the activation function
- The activation function (ReLU, sigmoid, tanh) is chosen independently
- Each neuron uses the same activation function type

**(c) Reduce overfitting** ✗

- Actually, increasing neurons typically **increases** overfitting risk
- More neurons = more parameters = higher capacity to memorize training data
- Regularization techniques (dropout, L2) are needed to combat this

**(d) Speed up training** ✗

- More neurons means MORE computations per forward/backward pass
- Training becomes **slower**, not faster
- More parameters to update means longer training time

**Clarification:**

The question is somewhat ambiguous:

- Adding neurons to a **layer** increases **width**
- Adding new **layers** with neurons increases **depth**

In practice, increasing neurons:

- **Pros:** Better representation capacity, can learn more complex functions
- **Cons:** More computation, higher overfitting risk, slower training

**More accurate statement:** Increasing the number of neurons helps to **increase the model's capacity and representational power**, though the question's answer (a) is the best choice among the given options.

---

## Question 6

**Backpropagation is primarily used for:**

- (a) Weight initialization
- (b) Weight update through gradient computation
- (c) Activation selection
- (d) Data preprocessing

### Solution:

**Answer: (b) Weight update through gradient computation**

**Explanation:**

Backpropagation (backward propagation of errors) is the fundamental algorithm for **training neural networks** by computing gradients of the loss function with respect to all network weights.

**Why Option (b) is Correct:**

Backpropagation performs two key functions:

1. **Gradient Computation:**

   - Calculates ∂L/∂w for every weight in the network
   - Uses the chain rule to propagate errors backward through layers
   - Example: ∂L/∂w₁ = ∂L/∂y · ∂y/∂z · ∂z/∂w₁

2. **Enables Weight Updates:**
   - Provides gradients needed for optimization algorithms (SGD, Adam, etc.)
   - Update rule: w_new = w_old - α · (∂L/∂w)
   - Without backpropagation, we couldn't train deep networks efficiently

**Why Other Options are Wrong:**

**(a) Weight initialization** ✗

- Weight initialization happens **before** training starts
- Common methods: Xavier, He initialization, random normal
- Backpropagation is not involved in initialization
- It only updates weights that are already initialized

**(c) Activation selection** ✗

- Activation functions (ReLU, sigmoid, tanh) are chosen during **architecture design**
- This is a manual decision by the network designer
- Backpropagation uses the derivative of activations but doesn't select them

**(d) Data preprocessing** ✗

- Preprocessing includes normalization, scaling, augmentation
- Happens **before** data enters the network
- Completely separate from backpropagation
- Example: StandardScaler, MinMaxScaler

**The Backpropagation Process:**

```
1. Forward Pass:
   Input → Layer 1 → Layer 2 → ... → Output → Loss

2. Backward Pass (Backpropagation):
   Loss → ∂L/∂w_n → ... → ∂L/∂w_2 → ∂L/∂w_1

3. Weight Update:
   w = w - α · ∂L/∂w (using computed gradients)
```

**Key Principle:**

Backpropagation uses the **chain rule** to efficiently compute gradients:

```
∂L/∂w₁ = (∂L/∂y) · (∂y/∂z) · (∂z/∂w₁)
```

This allows training of deep networks with millions of parameters in reasonable time.

---

## Question 7

**Write the equation for the tanh(x) activation function.**

### Solution:

#### **Hyperbolic Tangent (tanh) Activation Function:**

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

#### **Alternative Forms:**

**1. Using Exponentials:**

```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

**2. Using Sigmoid Function:**

```
tanh(x) = 2·σ(2x) - 1

where σ(x) = 1/(1 + e^(-x)) is the sigmoid function

Therefore:
tanh(x) = 2/(1 + e^(-2x)) - 1
```

**3. Simplified:**

```
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

#### **Properties:**

| Property          | Value                   |
| ----------------- | ----------------------- |
| **Range**         | (-1, 1)                 |
| **Domain**        | (-∞, ∞)                 |
| **Zero-centered** | Yes (outputs around 0)  |
| **Derivative**    | tanh'(x) = 1 - tanh²(x) |
| **At x=0**        | tanh(0) = 0             |
| **At x→∞**        | tanh(∞) = 1             |
| **At x→-∞**       | tanh(-∞) = -1           |

#### **Derivative:**

```
d/dx[tanh(x)] = 1 - tanh²(x) = sech²(x)
```

Or in terms of x:

```
tanh'(x) = 4e^(2x) / (e^(2x) + 1)²
```

#### **Key Values:**

```
tanh(0) = 0
tanh(1) ≈ 0.762
tanh(2) ≈ 0.964
tanh(5) ≈ 0.9999
tanh(-1) ≈ -0.762
```

#### **Graph Shape:**

```
        1 |     ________________
          |    /
          |   /
          |  /
    tanh  | /
          |/
    0 ----+-------------------- x
          |
          |
       -1 |________________

      S-shaped curve (sigmoid-like)
      Zero-centered (crosses at origin)
```

#### **Comparison with Sigmoid:**

| Feature       | tanh(x)                       | sigmoid(x)     |
| ------------- | ----------------------------- | -------------- |
| Range         | (-1, 1)                       | (0, 1)         |
| Zero-centered | Yes                           | No             |
| Formula       | (e^x - e^(-x))/(e^x + e^(-x)) | 1/(1 + e^(-x)) |
| Middle value  | 0                             | 0.5            |

#### **Usage in Neural Networks:**

- **Advantages over sigmoid:**
  - Zero-centered outputs (better gradient flow)
  - Stronger gradients (derivative range [0,1] vs sigmoid's [0,0.25])
- **Disadvantages:**

  - Still suffers from vanishing gradient problem for very large |x|
  - More expensive to compute than ReLU

- **Common use cases:**
  - RNN/LSTM cells (gate activations)
  - Hidden layers in shallow networks
  - When zero-centered outputs are beneficial

---

**END OF SOLUTIONS**

---

## Summary

This examination covered fundamental concepts in Deep Learning including:

- **Part A:**

  - Perceptron design and training (OR gate)
  - Classification vs Regression with loss functions
  - Forward pass and backpropagation in feedforward networks
  - Optimization algorithms (SGD vs Adam)

- **Part B:**
  - AI, ML, DL relationships
  - Gradient descent fundamentals
  - Neural network architecture concepts
  - Backpropagation purpose
  - Activation functions

**Key Takeaways:**

- Understanding of basic neural network operations
- Gradient-based optimization techniques
- Loss functions for different problem types
- Modern optimization improvements over classical methods

**Total Marks: 25**

Good luck with your examination! 🎓
