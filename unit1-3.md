# DEEP LEARNING - EXAM QUICK REVIEW (6 HOURS)

**Focus: Units 1-3 Important Topics Only**

---

## UNIT 1: MACHINE LEARNING FUNDAMENTALS

### 1. Classification vs Regression Algorithms

**Classification:**

- Predicts discrete categories/classes
- Output: Label (cat, dog, spam, not spam)
- Examples: Logistic Regression, SVM, Decision Trees, kNN
- Evaluation: Accuracy, Precision, Recall, F1-Score

**Regression:**

- Predicts continuous numerical values
- Output: Number (price, temperature, age)
- Examples: Linear Regression, Ridge, Lasso, Polynomial Regression
- Evaluation: MSE, RMSE, MAE, R²

**When to use:**

- Classification → Categorical target (e.g., disease yes/no)
- Regression → Numerical target (e.g., house price prediction)

---

### 2. Bias and Variance

**Bias:**

- Error from wrong assumptions in model
- High bias = Underfitting (model too simple)
- Example: Linear model for non-linear data

**Variance:**

- Error from sensitivity to training data fluctuations
- High variance = Overfitting (model too complex)
- Example: Very deep tree memorizes training noise

**Tradeoff:**
$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Goal:** Balance both (not too simple, not too complex)

**Visual:**

```
High Bias, Low Variance    → Underfitting (misses pattern)
Low Bias, High Variance    → Overfitting (memorizes noise)
Low Bias, Low Variance     → Good model ✓
```

---

### 3. Loss Functions

**Mean Squared Error (MSE) - Regression:**
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- Penalizes large errors heavily
- Sensitive to outliers

**Mean Absolute Error (MAE) - Regression:**
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

- Robust to outliers

**Cross-Entropy Loss - Classification:**
$$L_{CE} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

- For probability predictions
- Binary: $-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

**Hinge Loss - SVM:**
$$L = \max(0, 1 - y \cdot \hat{y})$$

- Margin-based loss

---

### 4. Perceptron Computation

**Algorithm:**

1. Initialize weights $w$ randomly
2. For each training example $(x, y)$:
   - Compute: $\hat{y} = \text{sign}(w^T x + b)$
   - If $\hat{y} \neq y$: Update $w = w + \eta(y - \hat{y})x$
3. Repeat until convergence

**Activation:**
$$\text{output} = \begin{cases} 1 & \text{if } w^T x + b > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Key Points:**

- Linear classifier (can only separate linearly separable data)
- Single layer neural network
- Learning rate $\eta$ controls step size

**Example:**

```
Input: x = [2, 3], Weights: w = [0.5, -1], Bias: b = 0.2
Activation: 0.5×2 + (-1)×3 + 0.2 = 1 - 3 + 0.2 = -1.8
Output: sign(-1.8) = -1 (class 0)
```

---

## UNIT 2: NEURAL NETWORKS BASICS

### 1. Activation Functions

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x)$$

- Most popular (fast, no vanishing gradient for x > 0)
- Gradient: 1 if x > 0, else 0
- Problem: Dead neurons (always outputs 0)

**Sigmoid:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Output range: (0, 1)
- Gradient: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Problem: Vanishing gradient (gradient → 0 for large |x|)

**Tanh (Hyperbolic Tangent):**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Gradient: $\tanh'(x) = 1 - \tanh^2(x)$

**Softmax (Output layer for multi-class):**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

- Converts logits to probabilities (sum = 1)

**When to use:**

- Hidden layers → ReLU (default choice)
- Binary output → Sigmoid
- Multi-class output → Softmax

---

### 2. Gradient Descent

**Concept:**

- Iteratively move weights opposite to gradient direction
- Goal: Minimize loss function

**Update Rule:**
$$w_{new} = w_{old} - \eta \cdot \nabla L(w)$$

Where:

- $\eta$ = learning rate (step size)
- $\nabla L(w)$ = gradient of loss w.r.t. weights

**Variants:**

**Batch Gradient Descent:**

- Use ALL training data per update
- Slow but stable

**Stochastic Gradient Descent (SGD):**

- Use 1 sample per update
- Fast but noisy

**Mini-batch Gradient Descent:**

- Use small batch (e.g., 32, 64 samples)
- Best balance (most common)

**Learning Rate:**

- Too large → Overshoots minimum
- Too small → Slow convergence
- Typical: 0.001, 0.01, 0.1

---

### 3. Gradient Computation (Backpropagation)

**Chain Rule:**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Where:

- $L$ = Loss
- $a$ = Activation
- $z$ = Pre-activation (weighted sum)
- $w$ = Weight

**Example: 2-layer network**

```
Input → Layer 1 → Layer 2 → Output → Loss

Forward:
x → z₁ = w₁x + b₁ → a₁ = σ(z₁) → z₂ = w₂a₁ + b₂ → ŷ = σ(z₂) → L

Backward:
∂L/∂w₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂w₂
∂L/∂w₁ = ∂L/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂w₁
```

**Key Gradients:**

Sigmoid: $\sigma'(x) = \sigma(x)(1-\sigma(x))$

ReLU: $\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$

Tanh: $\tanh'(x) = 1 - \tanh^2(x)$

---

## UNIT 3: CNN FUNDAMENTALS

### 1. Image Convolution Calculation

**Operation:**

- Slide kernel (filter) over input image
- Element-wise multiply kernel with input region
- Sum all products → single output value

**Example:**

Input (3×3):

```
[1  2  3]
[4  5  6]
[7  8  9]
```

Kernel (2×2):

```
[1  0]
[0  1]
```

**Calculation (stride=1, padding=0):**

Position (0,0):

```
[1  2]     [1  0]
[4  5]  ⊙  [0  1]  = 1×1 + 2×0 + 4×0 + 5×1 = 6
```

Position (0,1):

```
[2  3]     [1  0]
[5  6]  ⊙  [0  1]  = 2×1 + 3×0 + 5×0 + 6×1 = 8
```

Position (1,0):

```
[4  5]     [1  0]
[7  8]  ⊙  [0  1]  = 4×1 + 5×0 + 7×0 + 8×1 = 12
```

Position (1,1):

```
[5  6]     [1  0]
[8  9]  ⊙  [0  1]  = 5×1 + 6×0 + 8×0 + 9×1 = 14
```

**Output (2×2):**

```
[6   8]
[12  14]
```

---

### 2. Size of Output Feature Maps

**Formula:**
$$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$$

$$W_{out} = \frac{W_{in} - K + 2P}{S} + 1$$

Where:

- $H_{in}, W_{in}$ = Input height/width
- $K$ = Kernel size
- $P$ = Padding
- $S$ = Stride

**Common Cases:**

**Same padding (output = input size):**

```
224×224 input, 3×3 kernel, padding=1, stride=1
Output = (224 - 3 + 2×1)/1 + 1 = 224×224
```

**Valid padding (no padding):**

```
224×224 input, 3×3 kernel, padding=0, stride=1
Output = (224 - 3 + 0)/1 + 1 = 222×222
```

**Stride=2 (halve size):**

```
224×224 input, 3×3 kernel, padding=1, stride=2
Output = (224 - 3 + 2)/2 + 1 = 112×112
```

**MaxPooling 2×2 (halve size):**

```
224×224 input, 2×2 pool, stride=2
Output = (224 - 2)/2 + 1 = 112×112
```

---

### 3. Number of Parameters in CNN Layer

**Formula:**
$$\text{Parameters} = (K \times K \times C_{in} \times C_{out}) + C_{out}$$

Where:

- $K$ = Kernel size
- $C_{in}$ = Input channels
- $C_{out}$ = Output channels (number of filters)
- $C_{out}$ = Bias terms (one per filter)

**Examples:**

**Conv1 (RGB input):**

```
Input: 224×224×3
Filters: 64 filters of 3×3
Parameters = 3×3×3×64 + 64 = 1,728 + 64 = 1,792
```

**Conv2:**

```
Input: 112×112×64
Filters: 128 filters of 3×3
Parameters = 3×3×64×128 + 128 = 73,728 + 128 = 73,856
```

**VGG16 First Layer:**

```
Input: 224×224×3
Filters: 64 filters of 3×3
Parameters = 3×3×3×64 + 64 = 1,792
```

**AlexNet First Layer:**

```
Input: 227×227×3
Filters: 96 filters of 11×11
Parameters = 11×11×3×96 + 96 = 34,944
```

---

### 4. Object Detection

**Task:**

- Localize object (bounding box)
- Classify object (what is it)

**Output:**

- Bounding box: $(x, y, w, h)$ or $(x_{min}, y_{min}, x_{max}, y_{max})$
- Class probabilities: $[P(\text{cat}), P(\text{dog}), P(\text{car}), ...]$
- Confidence score: Probability that box contains object

**Difference from Classification:**

Classification:

- Input: Image
- Output: Class label only
- Example: "This is a cat"

Object Detection:

- Input: Image
- Output: Bounding box + Class label
- Example: "Cat at position (100, 150, 50, 80)"

**Popular Architectures:**

- YOLO (You Only Look Once) - Single shot detection
- R-CNN (Region-based CNN) - Two-stage detection
- SSD (Single Shot Detector)

**Metrics:**

- IoU (Intersection over Union): Overlap between predicted and ground truth box
- mAP (mean Average Precision): Average precision across all classes

**Example Output:**

```
Image: 416×416×3
Detection:
  Box 1: (120, 80, 200, 150), Class: Dog, Confidence: 0.95
  Box 2: (300, 200, 100, 120), Class: Car, Confidence: 0.87
```

---

## QUICK FORMULAS CHEATSHEET

**Loss Functions:**

- MSE: $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$
- MAE: $\frac{1}{N}\sum|y_i - \hat{y}_i|$
- Cross-Entropy: $-\sum y_i \log(\hat{y}_i)$

**Activations:**

- ReLU: $\max(0, x)$
- Sigmoid: $\frac{1}{1+e^{-x}}$
- Tanh: $\frac{e^x - e^{-x}}{e^x + e^{-x}}$
- Softmax: $\frac{e^{x_i}}{\sum e^{x_j}}$

**Gradient Descent:**

- Update: $w = w - \eta \nabla L$

**CNN Output Size:**

- $H_{out} = \frac{H_{in} - K + 2P}{S} + 1$

**CNN Parameters:**

- $\text{Params} = K \times K \times C_{in} \times C_{out} + C_{out}$

---

## EXAM TIPS (6 HOURS LEFT)

**Hour 1-2: Unit 1**

- Classification vs Regression (when to use)
- Bias-Variance (concept + visual understanding)
- Loss functions (formulas)
- Perceptron (algorithm + example calculation)

**Hour 3-4: Unit 2**

- Activation functions (formulas + when to use)
- Gradient descent (update rule + variants)
- Backpropagation (chain rule concept)
- Practice gradient computation for sigmoid/ReLU

**Hour 5-6: Unit 3**

- Convolution calculation (practice 2-3 examples)
- Output size formula (memorize + practice)
- Parameter counting (memorize formula + practice)
- Object detection (concept + difference from classification)

**Focus Areas:**
✓ Formulas (write them 5 times each)
✓ Numerical examples (do 3 practice problems per topic)
✓ Key differences (classification vs regression, bias vs variance, etc.)
✓ When to use what (activation functions, loss functions)

**Don't:**
✗ Read detailed theory (no time!)
✗ Try to understand everything deeply (exam mode!)
✗ Worry about advanced topics not in image

---

**END OF QUICK REVIEW - GOOD LUCK! 🎯**
