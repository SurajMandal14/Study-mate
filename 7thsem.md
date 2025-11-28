# Deep Learning Exam - Answer Sheet

## CSE 457 - Deep Learning Preparation Guide

**Subject:** Deep Learning  
**Exam Date:** December 8-11, 2025  
**Total Marks:** 20 (Section A: 10 marks, Section B: 10 marks)  
**Duration:** 60 minutes

---

## 📚 SECTION A: Multiple Choice Questions (10 × 1M = 10 Marks)

### **Question 1:** Computational Graph Gradient

**Question:** In a computational graph for the equation g = (x + y) \* z, what is the gradient of g with respect to z after a forward pass where x=1, y=2, z=3?

**Options:**
a) 1  
b) 3  
c) 5  
d) 9

**Answer: c) 5**

**Simple Explanation:**
Think of a computational graph like a recipe where you do math step by step:

- First, we add: x + y = 1 + 2 = 3
- Then, we multiply: g = 3 _ z = 3 _ 3 = 9

Now, the gradient means "how much does g change when z changes a tiny bit?"

- If z increases by 1, g = (x + y) _ z becomes (1 + 2) _ 4 = 12
- So gradient = (x + y) = 3 + 2 = **5**

**What is a gradient?** It's like asking "if I push this button (z) a little, how much does the output (g) move?"

---

### **Question 2:** Predicting House Prices - ML Technique

**Question:** You are building a model to predict house prices. This is an example of:

**Options:**
a) Classification  
b) Regression  
c) Clustering  
d) Dimensionality Reduction

**Answer: b) Regression**

**Simple Explanation:**
Imagine you're a real estate agent trying to guess house prices:

- **Regression** = Predicting a NUMBER (like price: $200,000, $350,000, etc.)
- **Classification** = Predicting a CATEGORY (like: "expensive" or "cheap")
- **Clustering** = Grouping similar things together (like finding neighborhoods)
- **Dimensionality Reduction** = Making complex data simpler

Since house prices are continuous numbers (can be any value), this is **regression**.

---

### **Question 3:** TensorFlow Code Understanding

**Question:** The TensorFlow code `tf.constant([5,2, name="x"], dtype=tf.float32)` creates a:

**Options:**
a) Variable tensor during training  
b) Placeholder for feeding data later  
c) Constant tensor with a value of 5.2  
d) Operation node in the graph named "x"

**Answer: c) Constant tensor with a value of 5.2**

**Simple Explanation:**
Think of TensorFlow like building with LEGO blocks:

- **Constant** = A block that NEVER changes (like writing in pen)
- **Variable** = A block that CAN change during learning (like writing in pencil)
- **Placeholder** = An empty box you'll fill later

Here, `tf.constant([5,2, name="x"])` creates a fixed number 5.2 that won't change. The `name="x"` is just a label, like putting a sticker on it.

---

### **Question 4:** Model Overfitting Problem

**Question:** A model performs exceptionally well on training data but poorly on unseen test data. This is a classic sign of:

**Options:**
a) Underfitting  
b) High Bias  
c) Overfitting  
d) Low Variance

**Answer: c) Overfitting**

**Simple Explanation:**
Imagine studying for an exam:

- **Overfitting** = You memorize ALL the practice questions perfectly, but fail the real exam because you didn't learn the concepts (just memorized answers)
- **Underfitting** = You barely studied, so you fail both practice and real exam
- **Good fit** = You understand the concepts, so you do well on both

When a model is TOO good on training data but bad on new data, it "memorized" instead of "learned" = **Overfitting**

---

### **Question 5:** Gradient Descent - Weight Update

**Question:** In the context of gradient descent, the learning rate:

**Options:**
a) Is the initial value of the weights  
b) Scales the magnitude of the weight update  
c) Determines the number of epochs  
d) Is the final loss value

**Answer: b) Scales the magnitude of the weight update**

**Simple Explanation:**
Think of learning like walking down a hill to find the bottom:

- **Gradient** = The direction of the slope (which way is down)
- **Learning rate** = How BIG your steps are

If learning rate is:

- **Too big** = You take huge steps and might jump over the bottom
- **Too small** = You take tiny steps and take forever to reach the bottom
- **Just right** = You reach the bottom efficiently

Learning rate controls HOW MUCH you change the weights each time.

---

### **Question 6:** Mean Squared Error (MSE) Use Case

**Question:** The Mean Squared Error (MSE) loss function is most appropriate for:

**Options:**
a) A binary classification (cat vs. dog)  
b) A multi-class classification problem (MNIST digits)  
c) A regression problem (predicting temperature)  
d) An unsupervised clustering problem

**Answer: c) A regression problem (predicting temperature)**

**Simple Explanation:**
MSE = Mean Squared Error = Average of (Prediction - Actual)²

Think of it like measuring how wrong your guesses are:

- **For numbers (temperature, price, age)** → Use MSE because it measures distance
  - If actual = 25°C, predicted = 23°C, error = (25-23)² = 4
- **For categories (cat/dog, yes/no)** → Use different loss functions because categories aren't numbers

MSE works best when predicting **continuous numbers** like temperature.

---

### **Question 7:** Primary Purpose of TensorFlow

**Question:** What is the primary purpose of TensorFlow?

**Options:**
a) To write TensorFlow code  
b) To provide a high-level API for building models  
c) To visualize data and monitor the training  
d) To optimize hyperparameters automatically

**Answer: b) To provide a high-level API for building models**

**Simple Explanation:**
TensorFlow is like a **toolbox** for building AI:

- **TensorFlow** = The main construction kit with all the tools
- **API (Application Programming Interface)** = Pre-made parts you can use (like LEGO instructions)

Instead of building everything from scratch (doing all the math manually), TensorFlow gives you ready-made functions to:

- Create neural networks
- Train models
- Make predictions

It's like using a microwave (TensorFlow) instead of building one from scratch!

---

### **Question 8:** tf.keras API Description

**Question:** The tf.keras API is best described as:

**Options:**
a) A low-level API for building custom training loops  
b) A high-level API for quickly building and training models  
c) A software acceleration library  
d) A data preprocessing toolkit

**Answer: b) A high-level API for quickly building and training models**

**Simple Explanation:**
Think of building a house:

- **Low-level API** = You need to make your own bricks, mix cement, everything from scratch (HARD)
- **High-level API** = You get pre-made walls, doors, windows that snap together (EASY)

**tf.keras** is the EASY way to build neural networks:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),  # Pre-made layer, just plug in!
    tf.keras.layers.Dense(10)
])
```

It's like assembling IKEA furniture with instructions, not building furniture from trees!

---

### **Question 9:** Linear Regression - Residual

**Question:** In a linear regression model y = b0 + b1\*x, the difference between the actual value and the predicted value for a data point is called the:

**Options:**
a) Gradient  
b) Residual  
c) Derivative  
d) Coefficient

**Answer: b) Residual**

**Simple Explanation:**
Imagine you're drawing a line through dots on a graph:

- **Predicted value** = Where your line says the point SHOULD be
- **Actual value** = Where the point REALLY is
- **Residual** = The gap between them (the mistake)

Example:

- Your line predicts: y = 10
- Actual point is: y = 12
- Residual = 12 - 10 = 2 (you were off by 2)

**Residual = Error = How wrong you were**

Think of it like archery: Residual is how far your arrow is from the bullseye!

---

### **Question 10:** Chain Rule in Calculus

**Question:** The chain rule in calculus is the fundamental principle behind:

**Options:**
a) Forward propagation  
b) Computational graph creation  
c) The tf.function decorator  
d) Backpropagation

**Answer: d) Backpropagation**

**Simple Explanation:**
The **chain rule** is a math trick for working backwards through connected things.

Imagine a factory assembly line:

1. Raw material → Machine A → Part 1
2. Part 1 → Machine B → Part 2
3. Part 2 → Machine C → Final product

**Backpropagation** = Working BACKWARDS to find problems:

- Final product is wrong → Check Machine C
- Machine C input (Part 2) is wrong → Check Machine B
- Machine B input (Part 1) is wrong → Check Machine A

**Chain rule** helps calculate how much EACH machine contributed to the final error, by "chaining" the effects backwards.

In neural networks:

- Forward propagation = Going through layers to make a prediction
- **Backpropagation** = Going BACKWARDS to adjust each layer based on the error

---

## 📝 SECTION B: Descriptive Questions (2Q × 5M = 10 Marks)

### **Question 11:** Forward Pass and Backward Pass (Backpropagation)

**Question:** Describe the steps involved in the forward pass and backward pass (backpropagation) for a simple computational graph g = a\*b + c. Use sample values a=2, b=3, c=4 to demonstrate the forward pass and then calculate the gradients ∂g/∂a, ∂g/∂b, and ∂g/∂c for the backward pass.

---

**ANSWER:**

#### **Understanding the Problem (For Beginners):**

Think of this like a recipe where we:

1. **Forward Pass** = Follow the recipe to cook (calculate the output)
2. **Backward Pass** = Figure out how each ingredient affects the final taste (calculate gradients)

**The equation:** g = a\*b + c

---

#### **FORWARD PASS (Going Forward - Making the Prediction):**

**What is Forward Pass?**  
Forward pass means calculating the output by going step-by-step through the operations.

**Steps:**

**Step 1:** Start with input values

- a = 2
- b = 3
- c = 4

**Step 2:** Calculate intermediate result (multiplication)

- Let's call the multiplication result "temp"
- temp = a _ b = 2 _ 3 = **6**

**Step 3:** Calculate final output (addition)

- g = temp + c = 6 + 4 = **10**

**Result:** g = 10

**Visual Representation:**

```
a=2 ────┐
        ├──→ [×] → temp=6 ──┐
b=3 ────┘                   ├──→ [+] → g=10
                            │
c=4 ────────────────────────┘
```

---

#### **BACKWARD PASS (Going Backward - Finding Gradients):**

**What is Backward Pass?**  
Backward pass (backpropagation) means working BACKWARDS to find out "how much does each input affect the output?"

**What are Gradients?**  
Gradients tell us: "If I change this input by 1, how much does the output change?"

---

**The Goal:** Find ∂g/∂a, ∂g/∂b, and ∂g/∂c

**Step-by-Step Calculation:**

**Step 1: ∂g/∂c (How does c affect g?)**

Looking at equation: g = a\*b + c

- If c increases by 1, g increases by 1
- **∂g/∂c = 1**

**Simple Explanation:** c is just added directly, so it has a 1-to-1 effect.

---

**Step 2: ∂g/∂temp (How does temp affect g?)**

Looking at: g = temp + c

- If temp increases by 1, g increases by 1
- **∂g/∂temp = 1**

---

**Step 3: ∂g/∂a (How does a affect g?)**

We need to use the **chain rule** here:

- a affects temp (through multiplication)
- temp affects g (through addition)

**Chain Rule:** ∂g/∂a = ∂g/∂temp × ∂temp/∂a

Calculate ∂temp/∂a:

- temp = a \* b
- If a increases by 1, temp increases by b
- ∂temp/∂a = b = 3

Therefore:

- **∂g/∂a = ∂g/∂temp × ∂temp/∂a = 1 × 3 = 3**

**Simple Explanation:** When 'a' increases by 1, it gets multiplied by b (which is 3), so g increases by 3.

---

**Step 4: ∂g/∂b (How does b affect g?)**

Similarly:

- b affects temp (through multiplication)
- temp affects g (through addition)

**Chain Rule:** ∂g/∂b = ∂g/∂temp × ∂temp/∂b

Calculate ∂temp/∂b:

- temp = a \* b
- If b increases by 1, temp increases by a
- ∂temp/∂b = a = 2

Therefore:

- **∂g/∂b = ∂g/∂temp × ∂temp/∂b = 1 × 2 = 2**

**Simple Explanation:** When 'b' increases by 1, it gets multiplied by a (which is 2), so g increases by 2.

---

#### **SUMMARY TABLE:**

| **Forward Pass**  | **Backward Pass (Gradients)** |
| ----------------- | ----------------------------- |
| a = 2             | ∂g/∂a = 3                     |
| b = 3             | ∂g/∂b = 2                     |
| c = 4             | ∂g/∂c = 1                     |
| temp = a\*b = 6   |                               |
| g = temp + c = 10 |                               |

---

#### **Verification (Checking Our Answer):**

Let's test if our gradients are correct:

**Test ∂g/∂a = 3:**

- Original: a=2, g=10
- Increase a by 1: a=3, g = 3\*3 + 4 = 13
- Change in g = 13-10 = 3 ✓ **Correct!**

**Test ∂g/∂b = 2:**

- Original: b=3, g=10
- Increase b by 1: b=4, g = 2\*4 + 4 = 12
- Change in g = 12-10 = 2 ✓ **Correct!**

**Test ∂g/∂c = 1:**

- Original: c=4, g=10
- Increase c by 1: c=5, g = 2\*3 + 5 = 11
- Change in g = 11-10 = 1 ✓ **Correct!**

---

#### **Why is this Important?**

In neural networks:

- **Forward pass** = Making predictions
- **Backward pass** = Learning from mistakes

Gradients tell the network HOW to adjust each parameter to reduce errors!

---

### **Question 12:** Mean Absolute Error vs Mean Squared Error

**Question:** Compare and contrast the Mean Absolute Error (L1 Loss) and Mean Squared Error (L2 Loss) functions. Explain with an example scenario why one might be preferred over the other.

---

**ANSWER:**

#### **Understanding Loss Functions (For Beginners):**

**What is a Loss Function?**  
A loss function measures HOW WRONG your predictions are. It's like a report card for your model.

Think of it like playing darts:

- Your throw = Prediction
- The bullseye = Actual target
- Loss = How far you are from the bullseye

---

#### **1. MEAN ABSOLUTE ERROR (MAE / L1 Loss)**

**Formula:**
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**In Simple Terms:**
$$MAE = \text{Average of } |Actual - Predicted|$$

**What does it mean?**

- Take the absolute difference (ignore negative signs)
- Average all the differences

**Example:**
Let's say you're predicting house prices (in thousands):

| House | Actual Price | Predicted Price | Error | Absolute Error |
| ----- | ------------ | --------------- | ----- | -------------- |
| 1     | 300          | 290             | -10   | 10             |
| 2     | 250          | 270             | +20   | 20             |
| 3     | 400          | 395             | -5    | 5              |

$$MAE = \frac{10 + 20 + 5}{3} = \frac{35}{3} = 11.67$$

**Simple Explanation:** On average, you're off by $11,670

---

#### **2. MEAN SQUARED ERROR (MSE / L2 Loss)**

**Formula:**
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**In Simple Terms:**
$$MSE = \text{Average of } (Actual - Predicted)^2$$

**What does it mean?**

- Calculate the difference
- SQUARE it (multiply by itself)
- Average all the squared differences

**Same Example:**

| House | Actual Price | Predicted Price | Error | Squared Error |
| ----- | ------------ | --------------- | ----- | ------------- |
| 1     | 300          | 290             | -10   | 100           |
| 2     | 250          | 270             | +20   | 400           |
| 3     | 400          | 395             | -5    | 25            |

$$MSE = \frac{100 + 400 + 25}{3} = \frac{525}{3} = 175$$

---

#### **COMPARISON TABLE:**

| **Aspect**                  | **MAE (L1 Loss)**          | **MSE (L2 Loss)**                      |
| --------------------------- | -------------------------- | -------------------------------------- |
| **Formula**                 | Average of absolute errors | Average of squared errors              |
| **Calculation**             | \|Actual - Predicted\|     | (Actual - Predicted)²                  |
| **Units**                   | Same as original data      | Squared units                          |
| **Sensitivity to Outliers** | Less sensitive             | MORE sensitive (punishes big errors)   |
| **Mathematical Property**   | Not differentiable at 0    | Smooth, differentiable everywhere      |
| **Gradient**                | Constant                   | Proportional to error                  |
| **Use Case**                | When outliers are present  | When you want to penalize large errors |

---

#### **KEY DIFFERENCES EXPLAINED SIMPLY:**

**1. How They Treat Errors:**

**MAE:** Treats all errors equally

- Error of 10 contributes 10 to the loss
- Error of 20 contributes 20 to the loss

**MSE:** Punishes big errors MORE

- Error of 10 contributes 100 to the loss
- Error of 20 contributes 400 to the loss (20²)

**Think of it like:**

- **MAE** = Teacher who deducts 1 point per wrong answer
- **MSE** = Teacher who deducts points exponentially (1 wrong = -1, 2 wrong = -4, 3 wrong = -9)

---

**2. Sensitivity to Outliers (Unusual Values):**

**Example Scenario:** Predicting student test scores

**Normal predictions:** 85, 87, 83, 86, 84  
**Actual scores:** 86, 88, 82, 85, 83

Now imagine ONE outlier (mistake):
**Predicted:** 85, 87, 83, 86, **50** (way off!)  
**Actual:** 86, 88, 82, 85, 83

**MAE Calculation:**

- Errors: 1, 1, 1, 1, 33
- MAE = (1+1+1+1+33)/5 = 7.4

**MSE Calculation:**

- Squared errors: 1, 1, 1, 1, 1089
- MSE = (1+1+1+1+1089)/5 = 218.6

**Observation:** MSE is MUCH higher because it squares the big error (33² = 1089)!

---

#### **WHEN TO USE EACH:**

**Use MAE (L1 Loss) When:**

1. **You have outliers in your data**

   - Example: Predicting house prices where some mansions are EXTREMELY expensive
   - MAE won't let those extreme values dominate the training

2. **All errors should be treated equally**

   - Example: Measuring distance traveled - 10km error is just twice as bad as 5km error

3. **You want robustness**
   - MAE is more stable when data has weird values

**Real Example:** Predicting delivery times

- Most deliveries: 30-40 minutes
- Some outliers: 2 hours (traffic jam)
- Use MAE so the traffic jams don't mess up predictions for normal deliveries

---

**Use MSE (L2 Loss) When:**

1. **You want to penalize large errors heavily**

   - Example: Self-driving cars - Being off by 1 meter is okay, but 10 meters could be fatal!
   - MSE will work harder to fix big mistakes

2. **Your data has few outliers**

   - Normal, well-behaved data

3. **You need smooth gradients for optimization**
   - MSE has nice mathematical properties for training

**Real Example:** Predicting stock prices

- Being off by $1 is acceptable
- Being off by $50 is REALLY bad (could lose lots of money)
- Use MSE to heavily penalize large prediction errors

---

#### **CONCRETE EXAMPLE SCENARIO:**

**Scenario:** Predicting Hospital Patient Recovery Time

**Dataset:**

- Most patients: 5-7 days recovery
- A few patients: 30+ days (complications)

**If you use MSE:**

- The model will focus HEAVILY on those 30-day patients
- Might predict 10-15 days for everyone to reduce the huge squared errors
- Normal patients get BAD predictions

**If you use MAE:**

- The model treats all patients more equally
- Predicts 5-7 days for normal patients accurately
- Still somewhat accounts for complicated cases
- **Better choice for this scenario!**

---

**Scenario 2:** Predicting Rocket Landing Coordinates

**Dataset:**

- Need to land within 1 meter of target
- Being off by 10 meters = crash = disaster

**If you use MAE:**

- Treats 1-meter error and 10-meter error somewhat similarly
- Might not focus enough on reducing large errors

**If you use MSE:**

- 10-meter error contributes 100× more than 1-meter error
- Model works EXTRA hard to avoid large mistakes
- **Better choice for this scenario!**

---

#### **MATHEMATICAL INSIGHT:**

**Why MSE Penalizes More:**

Small error: 2

- MAE contribution: 2
- MSE contribution: 4 (2×2)
- Ratio: 2:1

Large error: 10

- MAE contribution: 10
- MSE contribution: 100 (10×10)
- Ratio: 10:1

**The larger the error, the MORE disproportionate the penalty in MSE!**

---

#### **SUMMARY:**

| **Situation**                           | **Choose** | **Reason**                            |
| --------------------------------------- | ---------- | ------------------------------------- |
| Data has outliers                       | MAE        | Won't be dominated by extreme values  |
| Need to avoid large errors at all costs | MSE        | Heavily penalizes big mistakes        |
| Want equal treatment of all errors      | MAE        | Linear penalty                        |
| Normal, clean data                      | MSE        | Standard choice, smooth optimization  |
| Robust prediction needed                | MAE        | More stable                           |
| Critical application (safety)           | MSE        | Forces model to minimize large errors |

---

**Final Thought:**  
Think of it like parenting:

- **MAE** = Consistent, fair punishment for all mistakes
- **MSE** = Small mistakes get gentle correction, big mistakes get serious consequences

Both work, but you choose based on what behavior you want to encourage!

---

## 🎯 ADDITIONAL QUESTIONS FROM SECOND IMAGE

### **Question 1:** Multicollinearity in Variables

**Question:** The problem where independent variables in regression are highly correlated is called:

**Options:**
a) Homoscedasticity  
b) Autocorrelation  
c) Multicollinearity  
d) Linearity

**Answer: c) Multicollinearity**

**Simple Explanation:**
Imagine you're trying to predict ice cream sales using:

- Temperature
- "How hot it is"
- "Heat level"

These three things are basically THE SAME! When predictors are too similar, it's called **multicollinearity**.

**Problem:** The model gets confused because it can't tell which variable is actually important.

**Example:**

- Using both "height in inches" AND "height in centimeters" to predict weight
- They're just different ways to measure the same thing!

---

### **Question 2:** TensorFlow Function for Creating Tensors

**Question:** Which TensorFlow function is used to create a tensor of a specific shape filled with ones?

**Options:**
a) tf.zeros  
b) tf.fill  
c) tf.ones  
d) tf.eye

**Answer: c) tf.ones**

**Simple Explanation:**

```python
# Create a 3x3 grid filled with 1s
tensor = tf.ones([3, 3])

# Result:
# [[1, 1, 1],
#  [1, 1, 1],
#  [1, 1, 1]]
```

Think of it like:

- `tf.ones` = Fill a box with 1s
- `tf.zeros` = Fill a box with 0s
- `tf.fill` = Fill a box with any number you want

---

### **Question 3:** Vanishing Gradient Problem

**Question:** The vanishing gradient problem is most associated with which type of activation function?

**Options:**
a) ReLU  
b) Sigmoid  
c) Leaky ReLU  
d) Softmax

**Answer: b) Sigmoid**

**Simple Explanation:**
**Vanishing gradient** = Gradients become SUPER tiny during backpropagation, so learning almost stops.

**Sigmoid function:** Squishes all numbers between 0 and 1

- Big positive number → ~1
- Big negative number → ~0
- Problem: Its gradient (slope) is very small for large inputs

**Think of it like:**

- Trying to climb a hill that's almost flat → You barely move (vanishing gradient)
- **ReLU** is like a ramp → Easy to climb! (better choice)

**Why it's a problem:**
In deep networks with many layers, gradients get multiplied:

- 0.1 × 0.1 × 0.1 × 0.1 = 0.0001 (VERY small!)
- Early layers barely learn anything

---

### **Question 4:** Accuracy Formula

**Question:** The formula Accuracy = (TP + TN) / (P + N) is used to evaluate:

**Options:**
a) A regression model  
b) A clustering model  
c) A classification model  
d) A reinforcement learning agent

**Answer: c) A classification model**

**Simple Explanation:**
This formula is for classification (sorting things into categories).

**Breaking down the formula:**

- **TP** = True Positives (correctly said "yes")
- **TN** = True Negatives (correctly said "no")
- **P** = Total Positives (actual "yes" cases)
- **N** = Total Negatives (actual "no" cases)

**Example:** Email spam detector

- TP = Correctly identified spam (50 emails)
- TN = Correctly identified not spam (40 emails)
- Total emails = 100
- Accuracy = (50 + 40) / 100 = 90%

---

### **Question 5:** Machine Learning Equation x = w \* m + b

**Question:** In the equation for a line y = w \* m + b in the context of machine learning, what are called:

**Options:**
a) Features and labels  
b) Weights and bias  
c) Inputs and outputs  
d) Gradients and derivatives

**Answer: b) Weights and bias**

**Simple Explanation:**
This is like a seesaw equation:

**y = w \* m + b**

- **w** = Weight (how much does m matter? Like the importance)
- **m** = Input/Feature (the data you have)
- **b** = Bias (the starting point, like a baseline)

**Real example:** Predicting exam score

- m = hours studied (input)
- w = 10 (weight: each hour gives you 10 points)
- b = 30 (bias: you start with 30 points even with 0 study - prior knowledge)
- y = 10 \* hours + 30

If you study 5 hours: y = 10\*5 + 30 = 80 points!

---

### **Question 6:** Keras High-Level API Backend

**Question:** Keras is a high-level API that runs on top of which backend(s)?

**Options:**
a) Only TensorFlow  
b) Only PyTorch  
c) TensorFlow, Theano, and CNTK  
d) Only JAX

**Answer: c) TensorFlow, Theano, and CNTK** _(Historical answer)_

**Note:** In 2025, Keras primarily runs on TensorFlow, but originally it supported multiple backends!

**Simple Explanation:**
Think of **Keras** like a universal remote control:

- It has simple buttons (high-level API)
- It can control different TVs (TensorFlow, Theano, CNTK)

You write simple Keras code once, and it can work with different engines underneath!

---

### **Question 7:** tf.GradientTape() API in TensorFlow

**Question:** The tf.GradientTape() API in TensorFlow is used for:

**Options:**
a) Recording audio data  
b) Automatically calculating derivatives/gradients  
c) Visualizing data distributions  
d) Exporting models to production

**Answer: b) Automatically calculating derivatives/gradients**

**Simple Explanation:**
`tf.GradientTape()` is like a **recording device** for math operations.

```python
with tf.GradientTape() as tape:
    # Do some calculations
    y = x * x

# Now calculate the gradient (derivative)
gradient = tape.gradient(y, x)
```

**Think of it like:**

- You're doing math on a video camera
- Later, you can "rewind" and see how each step affected the result
- This helps calculate gradients for backpropagation!

**Why "Tape"?** Like an old cassette tape that records and plays back!

---

### **Question 8:** TensorFlow Highlight

**Question:** Which of the following is a highlight of TensorFlow?

**Options:**
a) It can only run on CPUs  
b) It is primarily used for data visualization  
c) Its code can run on multiple platforms (CPU, GPU, TPU)  
d) It does not support neural networks

**Answer: c) Its code can run on multiple platforms (CPU, GPU, TPU)**

**Simple Explanation:**
TensorFlow is like a video game that can run on:

- **CPU** = Regular computer processor (slower for AI)
- **GPU** = Graphics card (MUCH faster for AI - like a sports car)
- **TPU** = Special AI chips made by Google (fastest - like a rocket!)

**Why this matters:**

- You can write code once
- Run it on your laptop (CPU)
- Then run the same code on powerful servers (GPU/TPU) for faster training

---

### **Question 9:** False Positive in Binary Classification

**Question:** A False Positive in a binary classification model for spam detection occurs when:

**Options:**
a) A legitimate email is correctly classified as legitimate  
b) A spam email is correctly classified as spam  
c) A legitimate email is incorrectly classified as spam  
d) A spam email is incorrectly classified as legitimate

**Answer: c) A legitimate email is incorrectly classified as spam**

**Simple Explanation:**
**False Positive** = Model says "YES" but should have said "NO"

**In spam detection:**

- False Positive = Good email goes to spam folder (BAD!)
- You miss important emails!

**Memory trick:**

- **False** = Wrong
- **Positive** = Said "yes" (it's spam)
- **False Positive** = Wrongly said "yes" (called good email spam)

**Types of errors:**

1. **True Positive** = Correctly caught spam ✓
2. **True Negative** = Correctly identified good email ✓
3. **False Positive** = Good email marked as spam ✗ (annoying!)
4. **False Negative** = Spam gets through ✗ (also bad!)

---

### **Question 10:** Splitting Data into Training, Validation, and Test Sets

**Question:** The process of splitting data into Training, Validation, and Test sets is crucial to:

**Options:**
a) Increase the speed of training  
b) Prevent overfitting and get an unbiased estimate of model performance  
c) Decrease the model's complexity  
d) Choose the learning rate

**Answer: b) Prevent overfitting and get an unbiased estimate of model performance**

**Simple Explanation:**
Think of preparing for three different exams:

1. **Training Set (60%)** = Practice problems you study from

   - Model learns from these

2. **Validation Set (20%)** = Quiz to check your progress

   - Use this to tune your model (adjust hyperparameters)
   - Like practice exams

3. **Test Set (20%)** = FINAL EXAM
   - Never seen before
   - Tells you how well you'll do in real world
   - Model never trains on this!

**Why split?**

- If you practice ONLY on the final exam questions, you memorize answers (overfitting)
- Need fresh questions to truly test understanding
- Validation helps you improve WITHOUT cheating on the test set

---

### **Question 11:** TensorFlow Code - Creating Tensors and Operations

**Question:** Write the TensorFlow code to create two constant tensors A (2x3) and B (3x2) with random values. Then, perform multiplication between them using both tf.matmul and the @ operator. Finally, print the result. Explain the bias-variance trade-off. Describe what a high-bias (underfit) model and a high-variance (overfit) model look like in a graph of data points and a regression line/curve. What are two techniques to combat each?

---

**ANSWER:**

#### **Part 1: TensorFlow Code**

```python
import tensorflow as tf
import numpy as np

# Create two constant tensors with random values
# A is 2x3 (2 rows, 3 columns)
A = tf.constant(np.random.rand(2, 3), dtype=tf.float32, name="Matrix_A")

# B is 3x2 (3 rows, 2 columns)
B = tf.constant(np.random.rand(3, 2), dtype=tf.float32, name="Matrix_B")

print("Matrix A (2x3):")
print(A.numpy())
print("\nMatrix B (3x2):")
print(B.numpy())

# Method 1: Using tf.matmul (matrix multiplication function)
result_matmul = tf.matmul(A, B)
print("\nResult using tf.matmul:")
print(result_matmul.numpy())

# Method 2: Using @ operator (shorthand for matrix multiplication)
result_operator = A @ B
print("\nResult using @ operator:")
print(result_operator.numpy())

# Verify both methods give same result
print("\nAre both results equal?", tf.reduce_all(result_matmul == result_operator).numpy())
```

**Expected Output:**

```
Matrix A (2x3):
[[0.234  0.567  0.891]
 [0.432  0.123  0.789]]

Matrix B (3x2):
[[0.345  0.678]
 [0.234  0.890]
 [0.567  0.123]]

Result using tf.matmul:
[[0.769  1.234]
 [0.543  0.876]]

Result using @ operator:
[[0.769  1.234]
 [0.543  0.876]]

Are both results equal? True
```

**Simple Explanation:**

**What is Matrix Multiplication?**
Think of it like a recipe combiner:

- Matrix A (2x3): 2 recipes, each needs 3 ingredients
- Matrix B (3x2): 3 ingredients, each has 2 properties
- Result (2x2): 2 recipes with 2 final properties

**Why both methods work:**

- `tf.matmul(A, B)` = Official function (clear and explicit)
- `A @ B` = Shortcut (modern Python way)
- Both do EXACTLY the same thing!

---

#### **Part 2: Bias-Variance Trade-off**

**What is Bias-Variance Trade-off?**  
It's the balance between two types of errors your model can make.

Think of it like playing darts:

---

**1. HIGH BIAS (Underfitting):**

**What is it?**  
Model is TOO SIMPLE. It doesn't capture the pattern in the data.

**Dart Analogy:**

- You throw darts, but you're not even aiming at the board
- All darts land far from the bullseye, in the same wrong area
- You're consistently wrong in the same way

**In Machine Learning:**

```
Actual data: Curves and wiggles
Your model: Straight line

Data:    /\  /\  /\
Model:   __________ (just a flat line!)
```

**Example:**  
Trying to predict house prices using ONLY "number of rooms"

- Ignores location, size, age, etc.
- Too simple!

**Visual Graph:**

```
   Price
    |     ● ●    ●
    |   ●   ● ●    ●
    |  ●  /        ●
    | ●  /         ●
    |___/_____________ Rooms
        (Straight line misses the curve)
```

**Characteristics:**

- Poor performance on training data
- Poor performance on test data
- Model is too simple

---

**2. HIGH VARIANCE (Overfitting):**

**What is it?**  
Model is TOO COMPLEX. It memorizes training data instead of learning patterns.

**Dart Analogy:**

- Your darts are all over the place
- Sometimes close to bullseye, sometimes far
- No consistent pattern
- If you throw at a slightly different board, you miss completely

**In Machine Learning:**

```
Actual data: Gentle curve
Your model: Wild zigzag through every point

Data:    ● ● ● ● ●
Model:   \/\/\/\/\/ (zigzags through every point!)
```

**Example:**  
Memorizing every single house and its price

- Sees a new house → Can't predict (hasn't memorized it)
- Doesn't understand the general pattern

**Visual Graph:**

```
   Price
    |     ●  ●    ●
    |   ●  \/  ●    ●
    |  ●  /  \/     ●
    | ●  /    \     ●
    |___/______\______ Rooms
        (Crazy curve touches every point)
```

**Characteristics:**

- Perfect performance on training data
- Poor performance on test data
- Model is too complex

---

**THE TRADE-OFF:**

```
Simple Model     →→→→→     Complex Model
(High Bias)              (High Variance)
Underfitting              Overfitting

Too Simple  →  SWEET SPOT  ← Too Complex
```

**Goal:** Find the middle ground!

---

#### **Techniques to Combat Bias and Variance:**

**COMBATING HIGH BIAS (Underfitting):**

**Technique 1: Increase Model Complexity**

- Add more features
- Use a more powerful model (deeper neural network)
- Add polynomial features

**Example:**

```python
# Instead of: price = w1*rooms + b
# Use: price = w1*rooms + w2*rooms² + w3*location + w4*size + b
```

**Simple Explanation:**  
Like upgrading from a tricycle to a bicycle - give your model more power!

---

**Technique 2: Reduce Regularization**

- Regularization makes model simpler
- If model is too simple, reduce/remove regularization
- Let model learn more complex patterns

**Example:**

```python
# Remove or reduce the regularization parameter
model.compile(optimizer='adam',
              loss='mse',
              # Reduce regularization
              kernel_regularizer=tf.keras.regularizers.l2(0.001))  # Smaller value
```

---

**COMBATING HIGH VARIANCE (Overfitting):**

**Technique 1: Get More Training Data**

- More examples help model learn general patterns
- Harder to memorize when you have lots of data

**Simple Explanation:**  
Like studying from 100 practice tests instead of 5 - you learn patterns, not just memorize answers!

---

**Technique 2: Regularization (L1, L2, Dropout)**

**L1/L2 Regularization:** Penalize complex models

```python
model.add(Dense(128,
                kernel_regularizer=tf.keras.regularizers.l2(0.01)))
```

**Dropout:** Randomly turn off some neurons during training

```python
model.add(Dropout(0.5))  # Turn off 50% of neurons randomly
```

**Simple Explanation:**  
Like practicing with one hand tied behind your back - forces you to learn the fundamentals!

---

**Technique 3: Early Stopping**

- Stop training when validation performance stops improving
- Don't let model over-train

```python
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10)
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          callbacks=[early_stop])
```

---

**Technique 4: Reduce Model Complexity**

- Use fewer layers
- Use fewer neurons per layer
- Remove some features

**Example:**

```python
# Complex model (prone to overfitting):
model = Sequential([
    Dense(512),
    Dense(256),
    Dense(128),
    Dense(64),
    Dense(1)
])

# Simpler model:
model = Sequential([
    Dense(64),
    Dense(32),
    Dense(1)
])
```

---

#### **SUMMARY TABLE:**

| **Problem**          | **Symptoms**                       | **Solutions**                                |
| -------------------- | ---------------------------------- | -------------------------------------------- |
| **High Bias**        | Poor training accuracy             | 1. Increase model complexity                 |
| (Underfitting)       | Poor test accuracy                 | 2. Add more features                         |
|                      | Model too simple                   | 3. Reduce regularization                     |
|                      |                                    | 4. Train longer                              |
| -------------------- | ---------------------------------- | -------------------------------------------- |
| **High Variance**    | Perfect training accuracy          | 1. Get more training data                    |
| (Overfitting)        | Poor test accuracy                 | 2. Add regularization (L1/L2/Dropout)        |
|                      | Large gap between train & test     | 3. Early stopping                            |
|                      | Model too complex                  | 4. Reduce model complexity                   |

---

**Real-World Analogy:**

**Studying for Exam:**

**High Bias (Underfitting):**

- You barely study
- Don't understand the concepts
- Fail both practice tests and real exam

**High Variance (Overfitting):**

- You memorize specific practice questions
- Ace practice tests
- Fail real exam (questions are slightly different)

**Good Balance:**

- You understand concepts deeply
- Do well on practice tests
- Do well on real exam!

---

**Visual Comparison:**

```
UNDERFITTING (High Bias):
Data: ● ● ● ● ● ● ● ●
Model: _____________
(Straight line, misses everything)

GOOD FIT:
Data: ● ● ● ● ● ● ● ●
Model: ╱‾‾‾╲_____╱‾‾
(Smooth curve, follows general trend)

OVERFITTING (High Variance):
Data: ● ● ● ● ● ● ● ●
Model: ╱\╱\╱\╱\╱\╱\╱
(Zigzag through every point)
```

---

**Key Takeaway:**  
Machine learning is about finding the **Goldilocks zone**:

- Not too simple (high bias)
- Not too complex (high variance)
- Just right! (good generalization)

---

## 🎯 BONUS: Quick Tips for Your Exam

### **Common Patterns to Remember:**

1. **Gradient = Direction of Change**

   - It's always about "how much does output change when input changes"

2. **Forward = Predict, Backward = Learn**

   - Forward pass: Make predictions
   - Backward pass: Fix mistakes

3. **Loss Functions:**

   - Regression (numbers) → MSE or MAE
   - Classification (categories) → Cross-entropy

4. **Overfitting vs Underfitting:**

   - Overfitting = Memorizing (too complex)
   - Underfitting = Not learning enough (too simple)

5. **TensorFlow Basics:**
   - `tf.constant` = Fixed values
   - `tf.Variable` = Trainable values
   - `tf.GradientTape` = Record operations for gradients

---

## 📚 Study Strategy for December 8-11:

Since you have no background, focus on:

1. **Understand concepts, not memorize**

   - Read each explanation above slowly
   - Try to explain to yourself in your own words

2. **Key formulas to know:**

   - MAE = Average |Actual - Predicted|
   - MSE = Average (Actual - Predicted)²
   - Chain rule for backpropagation

3. **Practice with examples:**

   - Work through the forward/backward pass example multiple times
   - Try changing the numbers

4. **Common keywords:**
   - Gradient, Loss, Epoch, Batch
   - Overfitting, Underfitting, Regularization
   - Forward pass, Backward pass, Backpropagation

---

## ✨ Final Encouragement:

You're learning one of the most exciting fields in computer science! Deep learning might seem overwhelming, but remember:

- Every expert started as a beginner
- Understanding concepts > Memorizing formulas
- Real learning happens when you can explain it simply (which this answer sheet helps with!)

**Good luck on your exam!** 🚀 You've got this! 💪

---

## 📋 ADDITIONAL PRACTICE QUESTIONS (CLA Format)

### **Section A: Multiple Choice Questions (10 Marks)**

---

### **Question 1:** Keras History Since 2017

**Question:** Since 2017, Keras has been:

**Options:**
a) Replaced by PyTorch  
b) The official high-level API of TensorFlow  
c) Deprecated  
d) Available only in Theano

**Answer: b) The official high-level API of TensorFlow**

**Simple Explanation:**
In 2017, TensorFlow adopted Keras as its official way to build neural networks!

Think of it like:

- **TensorFlow** = The car engine (powerful but complex)
- **Keras** = The easy-to-use steering wheel and pedals (simple interface)

Before 2017: Keras could work with multiple backends (TensorFlow, Theano, CNTK)  
After 2017: Keras became the official "face" of TensorFlow - `tf.keras`

**Why this matters:**

- You get TensorFlow's power with Keras's simplicity
- Most modern TensorFlow code uses `tf.keras` API
- It's the recommended way to build models in TensorFlow

---

### **Question 2:** Vanishing Gradient Problem

**Question:** The vanishing gradient problem occurs mainly in:

**Options:**
a) CNNs  
b) RNNs  
c) Decision Trees  
d) Linear Regression

**Answer: b) RNNs (Recurrent Neural Networks)**

**Simple Explanation:**
**RNNs** are neural networks that process sequences (like sentences, time series data).

**The Problem:**
Imagine passing a message through 100 people in a line:

- First person: "The cat sat on the mat"
- By person 100: "Something about an animal?"
- Information gets lost!

**In RNNs:**

- Processing long sequences (like long sentences)
- Gradients get multiplied many times during backpropagation
- 0.1 × 0.1 × 0.1 × ... = Very tiny number!
- Early parts of the sequence don't learn anything

**Example:**
Sentence: "The cat, which was sitting on the comfortable red mat near the window, meowed"

- RNN struggles to connect "cat" at the start with "meowed" at the end
- Gradient vanishes before reaching back to "cat"

**Solution:** LSTM (Long Short-Term Memory) and GRU networks were invented to solve this!

**Why CNNs don't have this problem as much:**

- CNNs process images layer by layer
- Usually not as deep as RNN sequences
- Use activation functions like ReLU that help

---

### **Question 3:** Best Loss Function for Regression with Outliers

**Question:** Which loss function is best for regression with outliers?

**Options:**
a) MSE  
b) MAE  
c) Hinge Loss  
d) Cross-Entropy

**Answer: b) MAE (Mean Absolute Error)**

**Simple Explanation:**
Remember from earlier:

- **MSE** = Squares errors → Outliers get HUGE penalty → Model focuses too much on them
- **MAE** = Absolute errors → Treats all errors more equally → Robust to outliers

**Example with Outliers:**
Predicting apartment rent:

- Normal: $1000, $1100, $1200, $1150, $1050
- Outlier: $5000 (penthouse with special features)

**With MSE:**

- Error for $5000 outlier gets squared → Dominates the loss
- Model tries too hard to fit the outlier
- Normal apartments get bad predictions

**With MAE:**

- Outlier contributes proportionally
- Model focuses on the majority (normal apartments)
- Better overall predictions

**When to use each:**

- **MAE** → Data has outliers (recommended here!)
- **MSE** → Clean data, want to penalize large errors heavily

---

### **Question 4:** TensorFlow Backpropagation Foundation

**Question:** In TensorFlow, backpropagation relies on:

**Options:**
a) Chain rule of differentiation  
b) Statistical correlation  
c) Random guessing  
d) Gradient boosting

**Answer: a) Chain rule of differentiation**

**Simple Explanation:**
We covered this earlier, but let's reinforce:

**Chain Rule** = Math rule for finding derivatives of nested functions

**Think of a sandwich:**

- Bread → Lettuce → Tomato → Cheese → Bread
- If you want to know how changing the lettuce affects the final taste
- You need to consider: Lettuce → Tomato → Cheese → Final taste
- **Chain rule** helps you multiply all these effects together!

**In Neural Networks:**

```
Input → Layer 1 → Layer 2 → Layer 3 → Output → Loss
```

**Backpropagation:**

```
Loss ← Layer 3 ← Layer 2 ← Layer 1 ← (How to adjust weights?)
```

**Chain rule calculates:**

- How much does Layer 1's weight affect Layer 2?
- How much does Layer 2 affect Layer 3?
- How much does Layer 3 affect Loss?
- **Multiply them together** to see how Layer 1's weight affects final Loss!

**Formula:**
$$\frac{\partial Loss}{\partial w_1} = \frac{\partial Loss}{\partial a_3} \times \frac{\partial a_3}{\partial a_2} \times \frac{\partial a_2}{\partial a_1} \times \frac{\partial a_1}{\partial w_1}$$

This is the **chain** in "chain rule"!

---

### **Question 5:** Gradient Descent Optimization

**Question:** Gradient descent optimizes a model by:

**Options:**
a) Increasing error  
b) Minimizing loss  
c) Randomly updating weights  
d) Removing bias

**Answer: b) Minimizing loss**

**Simple Explanation:**
**Gradient Descent** = Walking down a hill to find the lowest point

Imagine you're on a foggy mountain and want to reach the valley:

1. Look around and see which direction is downhill (gradient)
2. Take a step in that direction (update weights)
3. Repeat until you can't go any lower (minimum loss)

**In Machine Learning:**

- **Hill height** = Loss (how wrong your predictions are)
- **Your position** = Current weights
- **Goal** = Find the lowest point (minimum loss = best predictions)

**The Process:**

```
Start → Calculate loss → Find gradient → Update weights → Repeat
```

**Example:**

- Current loss: 100 (very bad!)
- Gradient says: "Move weights this direction to reduce loss"
- Update weights
- New loss: 80 (better!)
- Keep going...
- Final loss: 5 (good!)

**Why "descent"?**

- You're descending (going down) the loss curve
- Always trying to minimize (reduce) the loss

---

### **Question 6:** Linear Regression Error Term

**Question:** In linear regression, the error between actual and predicted values is called:

**Options:**
a) Bias  
b) Residual  
c) Gradient  
d) Slope

**Answer: b) Residual**

**Simple Explanation:**
We covered this earlier! Let's review:

**Residual** = Actual value - Predicted value = The leftover error

**Visual Example:**

```
Actual point: ●
                    ↕ ← This gap is the residual
Predicted line: ─────
```

**Example:**

- Student studies 5 hours
- Model predicts: 75% score
- Actual score: 80%
- **Residual** = 80 - 75 = 5 (model underestimated by 5 points)

**Why it's called "residual":**

- It's what "remains" after your prediction
- The leftover error you couldn't explain
- Like residue left after cleaning!

**In practice:**

- Small residuals = Good model
- Large residuals = Model needs improvement
- Pattern in residuals = Model is missing something

---

### **Question 7:** Precision Metric Calculation

**Question:** The metric Precision is calculated as:

**Options:**
a) TP / (TP + FP)  
b) TP / P  
c) (TP + TN) / (P + N)  
d) FP / (FP + TN)

**Answer: a) TP / (TP + FP)**

**Simple Explanation:**
**Precision** answers: "Of all the things I said were positive, how many were actually positive?"

**Think of a spam filter:**

- You mark 100 emails as spam
- 80 actually are spam (TP = True Positives)
- 20 are good emails (FP = False Positives - you were wrong!)
- **Precision** = 80 / (80 + 20) = 80 / 100 = 80%

**Memory Trick:**
**Precision** = How **precise** you are when you say "YES"

- TP = You said "spam" and it IS spam ✓
- FP = You said "spam" but it's NOT spam ✗
- Precision = TP / (TP + FP) = Correct "YES" / All "YES"

**Example Scenario:**
Medical test for disease:

- Test says 10 people have disease
- 9 actually have it (TP = 9)
- 1 doesn't have it (FP = 1, false alarm)
- **Precision** = 9/10 = 90%
- High precision = Few false alarms!

**Compare with Recall:**

- **Precision** = Of predicted positives, how many are correct?
- **Recall** = Of actual positives, how many did we catch?

---

### **Question 8:** Creating Constants in TensorFlow

**Question:** In TensorFlow, constants are created using:

**Options:**
a) tf.constant()  
b) tf.variable()  
c) tf.Tensor()  
d) tf.add()

**Answer: a) tf.constant()**

**Simple Explanation:**

```python
# Create a constant (never changes)
x = tf.constant([1, 2, 3, 4, 5])

# Create a constant with specific type
y = tf.constant(3.14, dtype=tf.float32)

# Create a matrix constant
matrix = tf.constant([[1, 2], [3, 4]])
```

**Think of it like:**

- **tf.constant()** = Writing in permanent marker (can't change)
- **tf.Variable()** = Writing in pencil (can erase and change during training)

**When to use each:**

- **Constants** → Fixed values (like input data, fixed hyperparameters)
  - Example: `learning_rate = tf.constant(0.01)`
- **Variables** → Values that change during training (like weights)
  - Example: `weights = tf.Variable(tf.random.normal([784, 10]))`

**Quick comparison:**

```python
# Constant - cannot change
a = tf.constant(5)
# a = 10  ← This will create NEW variable, not change existing

# Variable - can be updated
b = tf.Variable(5)
b.assign(10)  # ✓ This CHANGES the value to 10
```

---

### **Question 9:** Deploying Keras Models

**Question:** Keras models can be deployed using:

**Options:**
a) Only Google Colab  
b) TensorFlow Lite, Cloud, or Mobile  
c) RStudio  
d) SQL Databases

**Answer: b) TensorFlow Lite, Cloud, or Mobile**

**Simple Explanation:**
Once you train a Keras model, you can deploy (use) it in many places:

**1. TensorFlow Lite** → Mobile phones and embedded devices

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Now run on Android/iOS!
```

- Use case: Face recognition on your phone camera

**2. Cloud** → Deploy on servers (AWS, Google Cloud, Azure)

```python
# Save model
model.save('my_model.h5')
# Upload to cloud and serve via API
```

- Use case: Image classification service for millions of users

**3. Mobile** → Edge devices, IoT

- Use case: Smart camera that detects objects offline

**4. Web** → TensorFlow.js for browsers

```javascript
// Load model in JavaScript
const model = await tf.loadLayersModel("model.json");
```

- Use case: Run AI directly in web browser!

**Why flexibility matters:**

- Same model, different platforms
- Choose based on needs (speed, privacy, cost)
- Train once, deploy everywhere!

---

### **Question 10:** Backpropagation Computes

**Question:** Backpropagation is used to compute:

**Options:**
a) Predictions  
b) Gradients  
c) Residuals  
d) Probabilities

**Answer: b) Gradients**

**Simple Explanation:**
**Backpropagation** = The algorithm that calculates gradients

**Remember:**

- **Forward Pass** → Make predictions (input → output)
- **Backpropagation** → Calculate gradients (output ← input)
- **Gradient Descent** → Use those gradients to update weights

**The Process:**

```
1. Forward Pass: Calculate prediction and loss
   Input → Layer 1 → Layer 2 → Output → Loss = 50

2. Backpropagation: Calculate gradients
   ← How to change weights ← ← ← Loss = 50
   Gradient for Layer 2 = 0.5
   Gradient for Layer 1 = 0.3

3. Update Weights: Use gradients to improve
   New weights = Old weights - (learning_rate × gradient)
```

**Why it's called "back" propagation:**

- Propagate = Spread information
- Back = Going backwards through the network
- Start from output error, go back to input

**What gradients tell us:**

- Which direction to adjust each weight
- How much to adjust it
- To reduce the loss

**Without backpropagation:**

- Would need to try random weight changes (very slow!)
- Backpropagation is efficient because it uses calculus

---

## 📝 SECTION B: DESCRIPTIVE QUESTIONS (10 Marks)

---

### **Question B.1:** Define Machine Learning and Its Types

**Question:** Define Machine Learning. Define different types of Machine Learning and give one example for each.

---

**ANSWER:**

#### **What is Machine Learning?**

**Simple Definition:**
Machine Learning is teaching computers to learn from experience (data) without being explicitly programmed for every situation.

**Think of it like:**

- **Traditional Programming:** You write exact rules
  - "If temperature > 30°C, say 'It's hot'"
  - "If temperature < 15°C, say 'It's cold'"
- **Machine Learning:** Computer learns rules from examples
  - Show it 1000 weather examples with labels
  - It figures out the pattern by itself!

**Formal Definition:**
Machine Learning is a subset of Artificial Intelligence that enables computers to learn and improve from experience (data) without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make decisions with minimal human intervention.

**Key Characteristics:**

1. Learns from data
2. Improves with experience
3. Makes predictions or decisions
4. Adapts to new situations

---

#### **Types of Machine Learning:**

There are **three main types** of Machine Learning:

---

### **1. SUPERVISED LEARNING**

**What is it?**
Learning from labeled data (data with answers provided).

**Think of it like:**

- A teacher giving you practice problems WITH answer key
- You learn by seeing questions and their correct answers
- Then you can solve new, similar problems

**How it works:**

1. Give computer data with labels (input + correct output)
2. Computer learns the mapping from input to output
3. Test on new data to see if it learned correctly

**The Process:**

```
Training Data:
Input: Picture of cat → Label: "Cat"
Input: Picture of dog → Label: "Dog"
Input: Picture of cat → Label: "Cat"
... (thousands of examples)

Result: Model learns to recognize cats vs dogs

New Input: Picture of new cat → Prediction: "Cat" ✓
```

---

**Examples of Supervised Learning:**

**Example 1: Email Spam Detection**

- **Input:** Email content ("Win free money!", "Meeting tomorrow at 3pm")
- **Label:** Spam or Not Spam
- **Learning:** Model learns patterns of spam emails
- **Prediction:** New email → Classify as spam or not

**Example 2: House Price Prediction**

- **Input:** House features (size, location, bedrooms, age)
- **Label:** Actual selling price
- **Learning:** Model learns relationship between features and price
- **Prediction:** New house → Predict its price

**Example 3: Handwritten Digit Recognition**

- **Input:** Image of handwritten digit
- **Label:** The actual digit (0-9)
- **Learning:** Model learns patterns of each digit
- **Prediction:** New handwritten digit → Recognize which number

---

**Two Subtypes:**

**a) Classification** → Predict categories

- Examples: Spam/Not Spam, Cat/Dog, Disease/Healthy
- Output: Discrete labels

**b) Regression** → Predict numbers

- Examples: House price, temperature, stock price
- Output: Continuous values

---

### **2. UNSUPERVISED LEARNING**

**What is it?**
Learning from unlabeled data (no answers provided).

**Think of it like:**

- Given a box of different colored and shaped toys
- No one tells you how to organize them
- YOU decide to group them (by color, size, shape, etc.)
- Finding hidden patterns yourself!

**How it works:**

1. Give computer data WITHOUT labels
2. Computer finds patterns, structures, or groupings by itself
3. Discovers hidden relationships in data

**The Process:**

```
Data: Pictures of various animals (no labels)

Model discovers:
Group 1: All have 4 legs, fur, bark → "Dog-like"
Group 2: All have 4 legs, fur, meow → "Cat-like"
Group 3: All have 2 legs, feathers → "Bird-like"

You never told it what cats, dogs, or birds are!
```

---

**Examples of Unsupervised Learning:**

**Example 1: Customer Segmentation**

- **Input:** Customer data (age, income, purchase history, browsing behavior)
- **No labels** provided
- **Learning:** Model groups similar customers together
- **Result:**
  - Group 1: Young, low income, buys budget items → "Budget Shoppers"
  - Group 2: Middle-aged, high income, buys premium → "Premium Customers"
  - Group 3: Students, frequent online shopping → "Online Enthusiasts"
- **Use:** Target marketing differently for each group!

**Example 2: Anomaly Detection**

- **Input:** Credit card transactions
- **No labels** for fraud
- **Learning:** Model learns "normal" transaction patterns
- **Result:** Flags unusual transactions that don't fit the pattern
  - Normal: $50 grocery, $30 gas
  - Anomaly: $5000 electronics from foreign country → Flag for review!

**Example 3: Document Clustering**

- **Input:** Thousands of news articles
- **No topic labels**
- **Learning:** Model groups similar articles together
- **Result:**
  - Cluster 1: All about sports
  - Cluster 2: All about politics
  - Cluster 3: All about technology

---

**Common Algorithms:**

- **K-Means Clustering** → Group similar data points
- **Hierarchical Clustering** → Create tree of clusters
- **PCA (Principal Component Analysis)** → Reduce dimensions
- **Autoencoders** → Learn compressed representations

---

### **3. REINFORCEMENT LEARNING**

**What is it?**
Learning through trial and error by interacting with an environment and receiving rewards or penalties.

**Think of it like:**

- Training a dog with treats!
  - Dog does something good → Give treat (reward)
  - Dog does something bad → No treat (penalty)
  - Dog learns to repeat actions that get treats
  - Dog avoids actions that don't get treats

**How it works:**

1. Agent (learner) takes actions in environment
2. Environment gives feedback (reward or penalty)
3. Agent learns which actions lead to best rewards
4. Goal: Maximize total reward over time

**The Process:**

```
Game Example: Learning to play chess

Action: Move piece → Outcome: Lost piece → Reward: -10 (bad!)
Action: Capture opponent → Outcome: Gained advantage → Reward: +5 (good!)
Action: Checkmate → Outcome: Won game → Reward: +100 (excellent!)

Over thousands of games, learns which moves lead to winning!
```

---

**Examples of Reinforcement Learning:**

**Example 1: Game Playing (AlphaGo, Chess AI)**

- **Agent:** AI player
- **Environment:** Game board
- **Actions:** Possible moves
- **Rewards:**
  - Win game: +1
  - Lose game: -1
  - Neutral moves: 0
- **Learning:** Play millions of games, learn winning strategies
- **Result:** AI becomes expert player

**Example 2: Robot Navigation**

- **Agent:** Robot
- **Environment:** Room with obstacles
- **Actions:** Move forward, turn left, turn right, stop
- **Rewards:**
  - Reach goal: +100
  - Hit obstacle: -50
  - Each step: -1 (encourage efficiency)
- **Learning:** Try different paths, learn optimal route
- **Result:** Robot navigates efficiently avoiding obstacles

**Example 3: Recommendation Systems**

- **Agent:** Recommendation algorithm
- **Environment:** User interaction
- **Actions:** Suggest different movies/products
- **Rewards:**
  - User watches full movie: +10
  - User clicks: +5
  - User ignores: 0
  - User dislikes: -5
- **Learning:** Learn user preferences over time
- **Result:** Better personalized recommendations

---

**Key Components:**

- **Agent:** The learner
- **Environment:** The world agent interacts with
- **State:** Current situation
- **Action:** What agent can do
- **Reward:** Feedback from environment
- **Policy:** Agent's strategy for choosing actions

---

#### **COMPARISON TABLE:**

| **Type**                | **Supervised**                | **Unsupervised**            | **Reinforcement**               |
| ----------------------- | ----------------------------- | --------------------------- | ------------------------------- |
| **Data**                | Labeled (with answers)        | Unlabeled (no answers)      | Interactive (feedback)          |
| **Goal**                | Predict output for input      | Find hidden patterns        | Maximize rewards                |
| **Learning Method**     | Learn from examples           | Discover structure          | Learn from trial and error      |
| **Teacher/Feedback**    | Explicit labels               | No teacher                  | Reward signals                  |
| **Example Input**       | Cat image with "Cat" label    | Cat image (no label)        | Game state                      |
| **Example Output**      | "This is a cat"               | "These images are similar"  | "Take this action"              |
| **Use Cases**           | Classification, Regression    | Clustering, Dimensionality  | Game AI, Robotics, Optimization |
| **Real-World Examples** | Spam detection, Price predict | Customer segmentation       | Self-driving cars, Game bots    |
| **Human Analogy**       | Learning with answer key      | Organizing without guidance | Learning by doing (rewards)     |

---

#### **VISUAL COMPARISON:**

**Supervised Learning:**

```
Teacher: "This is a cat" → Student learns → Sees new animal → Says "Cat!" ✓
```

**Unsupervised Learning:**

```
No teacher → Student sees many animals → Groups them → "These look similar!"
```

**Reinforcement Learning:**

```
Try action → Get reward/penalty → Adjust strategy → Try again → Improve over time
```

---

#### **Which Type to Choose?**

**Choose Supervised Learning when:**

- You have labeled data
- You know what you want to predict
- Clear input-output relationship
- Example: Image classification, price prediction

**Choose Unsupervised Learning when:**

- No labels available
- Want to explore data structure
- Looking for hidden patterns
- Example: Customer segmentation, anomaly detection

**Choose Reinforcement Learning when:**

- Need to make sequential decisions
- Can simulate environment
- Goal is long-term optimization
- Example: Game AI, robot control, trading strategies

---

#### **SUMMARY:**

**Machine Learning** is teaching computers to learn from data and experience.

**Three Main Types:**

1. **Supervised** → Learn from labeled examples (like school with answer key)
2. **Unsupervised** → Find patterns without labels (like organizing toys yourself)
3. **Reinforcement** → Learn through rewards and penalties (like training a pet)

Each type has its strengths and is used for different problems!

---

### **Question B.2:** Calculate Classification Metrics

**Question:** Calculate Precision, Recall, Accuracy and F1-Score for binary classification problem where TP=50, FN=10, FP=5, TN=35.

---

**ANSWER:**

#### **Understanding the Problem (For Beginners):**

**What are these terms?**
Imagine a medical test for a disease:

- **TP (True Positive) = 50** → Test says "Disease" and person HAS disease ✓ Correct!
- **FN (False Negative) = 10** → Test says "No disease" but person HAS disease ✗ Missed!
- **FP (False Positive) = 5** → Test says "Disease" but person is HEALTHY ✗ False alarm!
- **TN (True Negative) = 35** → Test says "No disease" and person is HEALTHY ✓ Correct!

**Total people tested:** 50 + 10 + 5 + 35 = **100 people**

---

#### **Visual Representation:**

```
                    ACTUAL CONDITION
                Positive        Negative
              (Has Disease)   (Healthy)
            ┌──────────────┬──────────────┐
Predicted   │              │              │
Positive    │   TP = 50    │   FP = 5     │  → Precision: 50/(50+5)
(Test says  │   ✓ Correct  │   ✗ Wrong    │
 Disease)   │              │              │
            ├──────────────┼──────────────┤
Predicted   │              │              │
Negative    │   FN = 10    │   TN = 35    │
(Test says  │   ✗ Wrong    │   ✓ Correct  │
 Healthy)   │              │              │
            └──────────────┴──────────────┘
                   ↑
            Recall: 50/(50+10)
```

---

#### **Given Values:**

- **TP (True Positive)** = 50
- **FN (False Negative)** = 10
- **FP (False Positive)** = 5
- **TN (True Negative)** = 35
- **Total** = 100

---

### **1. PRECISION**

**What does it measure?**
"When I say YES, how often am I correct?"

**Formula:**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Think of it as:**

- Of all the people we said have disease (TP + FP)
- How many actually have it? (TP)

**Calculation:**
$$\text{Precision} = \frac{50}{50 + 5} = \frac{50}{55} = 0.909$$

**Answer: Precision = 0.909 or 90.9%**

**What does this mean?**

- When our model predicts "Disease", it's correct 90.9% of the time
- Only 9.1% are false alarms
- High precision = Few false positives = Trustworthy positive predictions

**Example:** If test says you have disease, there's 91% chance you actually have it!

---

### **2. RECALL (Also called Sensitivity or True Positive Rate)**

**What does it measure?**
"Of all the actual positive cases, how many did I catch?"

**Formula:**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Think of it as:**

- Of all people who actually have disease (TP + FN)
- How many did we detect? (TP)

**Calculation:**
$$\text{Recall} = \frac{50}{50 + 10} = \frac{50}{60} = 0.833$$

**Answer: Recall = 0.833 or 83.3%**

**What does this mean?**

- We detected 83.3% of all disease cases
- We missed 16.7% (those are the False Negatives)
- High recall = Few missed cases = We catch most positive cases

**Example:** Of 60 people with disease, we correctly identified 50!

---

### **3. ACCURACY**

**What does it measure?**
"Overall, how many predictions were correct?"

**Formula:**
$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

**Think of it as:**

- Of all predictions (both positive and negative)
- How many were correct?

**Calculation:**
$$\text{Accuracy} = \frac{50 + 35}{50 + 5 + 10 + 35} = \frac{85}{100} = 0.85$$

**Answer: Accuracy = 0.85 or 85%**

**What does this mean?**

- 85% of all predictions were correct
- 15% were wrong (FP + FN = 5 + 10 = 15)
- Overall performance measure

**Example:** Out of 100 people tested, we got 85 correct answers!

---

### **4. F1-SCORE**

**What does it measure?**
"A balanced measure of Precision and Recall"

**Formula:**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Think of it as:**

- Harmonic mean of Precision and Recall
- Balances both metrics
- Useful when you care about both false positives AND false negatives

**Calculation:**
$$F1 = 2 \times \frac{0.909 \times 0.833}{0.909 + 0.833}$$

$$F1 = 2 \times \frac{0.757}{1.742}$$

$$F1 = 2 \times 0.435 = 0.870$$

**Answer: F1-Score = 0.870 or 87.0%**

**What does this mean?**

- Balanced measure between precision and recall
- Good when both false positives and false negatives matter
- Higher F1 = Better overall performance

---

#### **SUMMARY TABLE:**

| **Metric**    | **Formula**                     | **Calculation**       | **Result** | **Percentage** |
| ------------- | ------------------------------- | --------------------- | ---------- | -------------- |
| **Precision** | TP / (TP + FP)                  | 50 / (50 + 5)         | 0.909      | 90.9%          |
| **Recall**    | TP / (TP + FN)                  | 50 / (50 + 10)        | 0.833      | 83.3%          |
| **Accuracy**  | (TP + TN) / (TP+FP+FN+TN)       | (50 + 35) / 100       | 0.850      | 85.0%          |
| **F1-Score**  | 2×(Precision×Recall)/(Prec+Rec) | 2×(0.909×0.833)/1.742 | 0.870      | 87.0%          |

---

#### **INTERPRETATION:**

**For this model:**

✓ **Precision = 90.9%** (HIGH)

- When model says "Disease", it's usually right
- Few false alarms
- Trustworthy positive predictions

✓ **Recall = 83.3%** (GOOD)

- Catches most disease cases
- Misses about 17% of cases
- Could be improved to catch more

✓ **Accuracy = 85%** (GOOD)

- Overall correct 85% of the time
- Solid general performance

✓ **F1-Score = 87%** (GOOD)

- Balanced performance
- Good trade-off between precision and recall

---

#### **WHEN EACH METRIC MATTERS:**

**Prioritize PRECISION when:**

- False positives are expensive/harmful
- Example: Spam filter (don't want good emails in spam)
- Example: Court verdict (don't convict innocent person)

**Prioritize RECALL when:**

- False negatives are dangerous
- Example: Cancer detection (don't want to miss cancer)
- Example: Fraud detection (catch all fraudulent transactions)

**Use ACCURACY when:**

- Classes are balanced
- Both errors are equally important
- Simple overall measure needed

**Use F1-SCORE when:**

- Need balance between precision and recall
- Classes might be imbalanced
- Both types of errors matter

---

#### **REAL-WORLD CONTEXT:**

**Medical Test Example:**
Our model:

- Correctly identifies 50/55 disease predictions (Precision = 90.9%)
- Catches 50/60 actual disease cases (Recall = 83.3%)
- Gets 85/100 total predictions right (Accuracy = 85%)

**Is this good?**

- **Depends on the disease!**
- For common cold: This is excellent!
- For deadly disease: 83.3% recall means we're missing 10 people who have it - need improvement!

---

#### **CONFUSION MATRIX VISUALIZATION:**

```
                Confusion Matrix

                Predicted
                P       N
Actual  P      50      10     ← 60 actual positives
                      (Missed!)

        N       5      35     ← 40 actual negatives
             (False
              alarm!)

              55      45
          (Predicted  (Predicted
           Positive)   Negative)

Precision looks at this column ↓
              TP/(TP+FP)

Recall looks at this row →
              TP/(TP+FN)
```

---

#### **KEY TAKEAWAYS:**

1. **Precision** = Quality of positive predictions (fewer false alarms)
2. **Recall** = Quantity of positives caught (fewer missed cases)
3. **Accuracy** = Overall correctness (simple but can be misleading with imbalanced data)
4. **F1-Score** = Harmonic mean of Precision and Recall (balanced measure)

**The Trade-off:**

- High Precision often means Lower Recall (and vice versa)
- Be strict → Few false positives but miss some cases (High Precision, Low Recall)
- Be lenient → Catch all cases but more false alarms (Low Precision, High Recall)
- **F1-Score helps find the sweet spot!**

---

**Final Answer Summary:**

- **Precision** = 90.9%
- **Recall** = 83.3%
- **Accuracy** = 85.0%
- **F1-Score** = 87.0%

This model has good overall performance with high precision and decent recall!

---

_Created for exam preparation - December 2025_  
_Subject: CSE 457 - Deep Learning_  
_Focus: Beginner-friendly explanations with practical examples_
