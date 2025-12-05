# Deep Learning Mid-Semester Examination

## B.Tech CSE - Final Year

**Student Name:** _[Your Name]_  
**Roll Number:** _[Your Roll Number]_  
**Date:** December 5, 2025  
**Total Marks:** 20 (Section A: 10 Marks, Section B: 10 Marks)

---

## Section A: Multiple Choice Questions (10 × 1M = 10 Marks)

### **Question 1**

In a computational graph for the equation g = (x + y) \* z, what is the gradient of g with respect to z after a forward pass where x = 1, y = 2, z = 3?

**Answer: b) 3**

**Explanation:**

- Forward pass: g = (1 + 2) _ 3 = 3 _ 3 = 9
- ∂g/∂z = (x + y) = (1 + 2) = 3
- The gradient represents how much g changes with respect to z, which is the sum (x + y).

---

### **Question 2**

You are building a model to predict house prices. This is an example of:

**Answer: c) Regression**

**Explanation:**
Predicting house prices is a regression problem because we're predicting a continuous numerical value (price). Classification deals with discrete categories, clustering groups similar data points, and dimensionality reduction reduces feature space.

---

### **Question 3**

The TensorFlow code `tf.constant(5.2, name="x", dtype=tf.float32)` creates a:

**Answer: c) Constant tensor with a value of 5.2**

**Explanation:**
`tf.constant()` creates an immutable tensor with a fixed value that cannot be changed during training. Variables (`tf.Variable`) can be updated, while this constant remains fixed at 5.2 throughout the execution.

---

### **Question 4**

A model performs exceptionally well on training data but poorly on unseen test data. This is a classic sign of:

**Answer: c) Overfitting**

**Explanation:**
Overfitting occurs when the model learns the training data too well, including its noise and peculiarities, but fails to generalize to new, unseen data. This is characterized by high training accuracy but low test accuracy. Underfitting would show poor performance on both training and test data.

---

### **Question 5**

In the context of gradient descent, the learning rate:

**Answer: b) Scales the magnitude of the weight update**

**Explanation:**
The learning rate (α) controls how big the steps are when updating weights during gradient descent. The update rule is: weight_new = weight_old - α \* gradient. A larger learning rate means bigger steps, while a smaller one means smaller, more cautious steps.

---

### **Question 6**

The Mean Squared Error (MSE) loss function is most appropriate for:

**Answer: c) A regression problem (predicting temperature)**

**Explanation:**
MSE is designed for regression tasks where we predict continuous values. It calculates the average of squared differences between predicted and actual values. For classification problems, we typically use cross-entropy loss instead.

---

### **Question 7**

What is the primary purpose of TensorBoard?

**Answer: c) To visualize and monitor the training process and graph**

**Explanation:**
TensorBoard is a visualization toolkit that allows us to visualize metrics like loss and accuracy over time, examine the computational graph, view histograms of weights and gradients, and monitor the training process. It's an essential tool for debugging and understanding model behavior.

---

### **Question 8**

The _tf.keras_ API is best described as:

**Answer: b) A high-level API for quickly building and training models**

**Explanation:**
tf.keras provides a user-friendly, high-level interface for building neural networks. It abstracts away many low-level details and allows rapid prototyping with simple APIs like Sequential() and Model(). For more control, TensorFlow's low-level APIs can be used.

---

### **Question 9**

In a linear regression model y = b0 + b1\*x, the difference between the actual value and the predicted value for a data point is called:

**Answer: b) Residual**

**Explanation:**
The residual (or error) is the difference between the observed actual value and the predicted value: residual = y_actual - y_predicted. Minimizing the sum of squared residuals is the goal of linear regression.

---

### **Question 10**

The chain rule in calculus is the fundamental principle behind:

**Answer: d) Backpropagation**

**Explanation:**
Backpropagation uses the chain rule to compute gradients of the loss function with respect to each parameter in the network. The chain rule allows us to decompose the derivative of a composite function into the product of derivatives of its constituent functions, which is essential for propagating errors backward through the network layers.

---

## Section B: Descriptive Questions (2 × 5M = 10 Marks)

### **Question 11**

Describe the steps involved in the forward pass and backward pass (backpropagation) for a simple computational graph g = (a + b) \* c. Use sample values a = 2, b = 3, c = 4 to demonstrate the forward pass and then calculate the gradients ∂g/∂a, ∂g/∂b, and ∂g/∂c in the backward pass.

**Answer:**

#### **Forward Pass:**

The computational graph for g = (a + b) \* c can be broken down into intermediate steps:

1. **Given values:**

   - a = 2
   - b = 3
   - c = 4

2. **Intermediate computation:**

   - Let s = a + b
   - s = 2 + 3 = 5

3. **Final output:**
   - g = s \* c
   - g = 5 \* 4 = **20**

**Computational Graph Structure:**

```
    a(2)  b(3)
      \   /
       (+)
        |
       s(5)    c(4)
         \     /
          (*)
           |
          g(20)
```

#### **Backward Pass (Backpropagation):**

Starting from the output and working backward using the chain rule:

1. **Gradient of g with respect to itself:**

   - ∂g/∂g = 1 (starting point)

2. **Gradient with respect to c:**

   - g = s \* c
   - ∂g/∂c = s = **5**

   _Interpretation:_ If c increases by 1, g increases by 5 (the value of s).

3. **Gradient with respect to s (intermediate node):**

   - g = s \* c
   - ∂g/∂s = c = 4

4. **Gradient with respect to a:**

   - Using chain rule: ∂g/∂a = ∂g/∂s × ∂s/∂a
   - s = a + b, so ∂s/∂a = 1
   - ∂g/∂a = 4 × 1 = **4**

   _Interpretation:_ If a increases by 1, g increases by 4.

5. **Gradient with respect to b:**

   - Using chain rule: ∂g/∂b = ∂g/∂s × ∂s/∂b
   - s = a + b, so ∂s/∂b = 1
   - ∂g/∂b = 4 × 1 = **4**

   _Interpretation:_ If b increases by 1, g increases by 4.

#### **Summary of Results:**

| Variable | Gradient Value | Meaning                     |
| -------- | -------------- | --------------------------- |
| ∂g/∂a    | 4              | Rate of change of g w.r.t a |
| ∂g/∂b    | 4              | Rate of change of g w.r.t b |
| ∂g/∂c    | 5              | Rate of change of g w.r.t c |

**Verification:**

- Both a and b have the same gradient (4) because they contribute equally to the sum s
- c has a gradient of 5 because it multiplies the result of (a + b)
- These gradients would be used to update the parameters if this were a training step in a neural network

---

### **Question 12**

Compare and contrast the Mean Absolute Error (L1 Loss) and Mean Squared Error (L2 Loss) functions. Explain with an example scenario why one might be preferred over the other.

**Answer:**

#### **1. Mathematical Definitions:**

**Mean Absolute Error (MAE / L1 Loss):**

```
MAE = (1/n) × Σ|y_actual - y_predicted|
```

**Mean Squared Error (MSE / L2 Loss):**

```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

where n is the number of samples.

#### **2. Key Differences:**

| Aspect                       | L1 Loss (MAE)                           | L2 Loss (MSE)              |
| ---------------------------- | --------------------------------------- | -------------------------- |
| **Calculation**              | Sum of absolute differences             | Sum of squared differences |
| **Sensitivity to outliers**  | Less sensitive (robust)                 | Highly sensitive           |
| **Gradient**                 | Constant (±1)                           | Proportional to error      |
| **Optimization**             | Can be harder (non-differentiable at 0) | Smoother optimization      |
| **Error magnitude**          | Linear scale                            | Quadratic scale            |
| **Penalty for large errors** | Linear penalty                          | Exponential penalty        |

#### **3. Detailed Comparison:**

**a) Outlier Sensitivity:**

- **L1 Loss:** Treats all errors equally with linear penalty. Less affected by outliers.
- **L2 Loss:** Squares the errors, so large errors are penalized much more heavily than small ones. Very sensitive to outliers.

**b) Gradient Behavior:**

- **L1 Loss:** Has a constant gradient (sign of the error), which can make convergence less smooth.
- **L2 Loss:** Has a gradient proportional to the error, providing smoother convergence as the model approaches the optimum.

**c) Robustness:**

- **L1 Loss:** More robust to outliers and noisy data.
- **L2 Loss:** Can be significantly affected by outliers, potentially skewing the model.

#### **4. Example Scenario:**

**Scenario: Predicting Delivery Times for an E-commerce Platform**

Consider predicting delivery times where most deliveries take 2-5 days, but occasionally there are extreme delays (15-20 days) due to unavoidable circumstances like weather or customs issues.

**Sample Dataset:**

```
Actual:    [3, 4, 3, 5, 4, 18, 3, 4]  (days)
Predicted: [3.5, 4.2, 3.1, 5.3, 4.1, 5.0, 3.2, 4.0]  (days)
```

**Using L2 Loss (MSE):**

- Error for outlier: (18 - 5.0)² = 169
- Error for typical: (3 - 3.5)² = 0.25
- The outlier dominates the loss function (169 vs 0.25)
- Model will try very hard to fit the outlier, potentially degrading performance on typical cases

**Using L1 Loss (MAE):**

- Error for outlier: |18 - 5.0| = 13
- Error for typical: |3 - 3.5| = 0.5
- While the outlier still contributes more, the difference is less dramatic (13 vs 0.5)
- Model maintains better overall performance on typical cases

#### **5. When to Use Each:**

**Use L1 Loss (MAE) when:**

- Dataset contains significant outliers that you don't want to dominate training
- You want to treat all errors equally important
- Example: Predicting house prices in an area with a few luxury mansions among normal houses
- The target distribution is not Gaussian

**Use L2 Loss (MSE) when:**

- Outliers are rare and represent genuinely important errors to minimize
- You want large errors to be penalized heavily
- The optimization landscape needs to be smooth
- Example: Predicting temperature or sensor readings where large errors are critical
- The target distribution is approximately Gaussian

#### **6. Practical Consideration:**

In the delivery time prediction scenario, **L1 Loss would be preferred** because:

1. Extreme delays are rare and unavoidable
2. We want to optimize for typical delivery times (majority of cases)
3. A few outliers shouldn't drastically affect the model's performance on normal deliveries
4. The business cares more about consistent, reliable predictions for regular deliveries

Conversely, if we were predicting medical dosages or safety-critical measurements, **L2 Loss might be preferred** because large errors could be catastrophic and must be heavily penalized.

---

**End of Answer Sheet**

---

## Additional Notes:

- All answers are provided based on fundamental deep learning concepts covered in the course
- Mathematical derivations follow standard calculus and optimization principles
- Practical examples reflect real-world machine learning applications
- Concepts align with TensorFlow 2.x framework implementations
