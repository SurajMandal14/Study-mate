# Soft Computing End Semester Exam Solutions (Part 1)

**Subject:** Principles of Soft Computing (CSE 412)
**Date:** Dec 2024 Paper Solution

---

## Section A

### **Q1. Hebb Rule Classification (10 Marks)**

**Problem:**
Find the weights required to classify the patterns 'I' and 'O' using the Hebb rule.

- **Pattern 'I':** Target = 1
- **Pattern 'O':** Target = -1
- **Representation:** '+' = 1, Empty = -1.

**Solution:**

**Step 1: Represent the Input Patterns as Vectors**
We read the 3x3 grids row by row to create 9-dimensional vectors ($x_1$ to $x_9$).

**Pattern 1 (Class 'I', Target $y_1 = 1$):**
Grid:

```
+ + +  (1, 1, 1)
  +    (-1, 1, -1)  <-- Assuming empty squares are -1
+ + +  (1, 1, 1)
```

Vector $X_1 = [1, 1, 1, -1, 1, -1, 1, 1, 1]$

**Pattern 2 (Class 'O', Target $y_2 = -1$):**
Grid:

```
+ + +  (1, 1, 1)
+   +  (1, -1, 1)
+ + +  (1, 1, 1)
```

Vector $X_2 = [1, 1, 1, 1, -1, 1, 1, 1, 1]$

**Step 2: Initialize Weights and Bias**
Assume initial weights $W = [0, 0, 0, 0, 0, 0, 0, 0, 0]$ and Bias $b = 0$.

**Step 3: Apply Hebb Rule**
Formula: $W_{new} = W_{old} + X \cdot y$
Formula: $b_{new} = b_{old} + y$

**Update for Pattern 1 ('I'):**
$W_1 = W_0 + X_1 \cdot (1) = X_1$
$W_1 = [1, 1, 1, -1, 1, -1, 1, 1, 1]$
$b_1 = 0 + 1 = 1$

**Update for Pattern 2 ('O'):**
$W_{final} = W_1 + X_2 \cdot (-1)$
$W_{final} = W_1 - X_2$
$b_{final} = 1 + (-1) = 0$

**Calculation ($W_1 - X_2$):**

- $w_1 = 1 - 1 = 0$
- $w_2 = 1 - 1 = 0$
- $w_3 = 1 - 1 = 0$
- $w_4 = -1 - 1 = -2$
- $w_5 = 1 - (-1) = 2$
- $w_6 = -1 - 1 = -2$
- $w_7 = 1 - 1 = 0$
- $w_8 = 1 - 1 = 0$
- $w_9 = 1 - 1 = 0$

**Final Weights:** $W = [0, 0, 0, -2, 2, -2, 0, 0, 0]$
**Final Bias:** $b = 0$

---

### **Q2. Convolutional Neural Networks (10 Marks)**

#### **a) Purpose of Filters (Kernels) (3 Marks)**

Filters (or kernels) in CNNs are small matrices of weights that slide over the input image to detect specific features.

- **Feature Extraction:** They act as feature detectors. Low-level filters might detect edges (vertical, horizontal), curves, or colors. High-level filters in deeper layers detect complex patterns like eyes or wheels.
- **Spatial Hierarchy:** By convolving filters over the image, the network learns spatial hierarchies of patterns.
- **Parameter Sharing:** The same filter is used across the entire image, making the network translation invariant and reducing the number of parameters compared to fully connected networks.

#### **b) Convolution Calculation (7 Marks)**

**Given:**

- Input Image $I$ (6x6)
- Filter $K$ (3x3)
- Stride = 1, No Padding.

**Filter K:**

```
1  0  -1
1  0  -1
1  0  -1
```

_(This is a vertical edge detector. It subtracts the right column pixel values from the left column pixel values)._

**1) Perform Convolution (No Padding)**
Output Size Formula: $O = \frac{W - K}{S} + 1$
$O = \frac{6 - 3}{1} + 1 = 4$
The output will be a **4x4 matrix**.

**Calculations (Row by Row):**
Let's denote the 3x3 patch of the image as $P$. The operation is sum of element-wise product $P \cdot K$.
Since the middle column of K is 0, we just calculate: $\sum(\text{Left Col of P}) - \sum(\text{Right Col of P})$.

- **Row 1:**

  - (0,0): Patch $\begin{bmatrix}1&2&3\\4&1&0\\1&3&1\end{bmatrix} \rightarrow (1+4+1) - (3+0+1) = 6 - 4 = \mathbf{2}$
  - (0,1): Patch $\begin{bmatrix}2&3&0\\1&0&2\\3&1&2\end{bmatrix} \rightarrow (2+1+3) - (0+2+2) = 6 - 4 = \mathbf{2}$
  - (0,2): Patch $\begin{bmatrix}3&0&1\\0&2&3\\1&2&0\end{bmatrix} \rightarrow (3+0+1) - (1+3+0) = 4 - 4 = \mathbf{0}$
  - (0,3): Patch $\begin{bmatrix}0&1&2\\2&3&4\\2&0&1\end{bmatrix} \rightarrow (0+2+2) - (2+4+1) = 4 - 7 = \mathbf{-3}$

- **Row 2:**

  - (1,0): Patch $\begin{bmatrix}4&1&0\\1&3&1\\0&1&2\end{bmatrix} \rightarrow (4+1+0) - (0+1+2) = 5 - 3 = \mathbf{2}$
  - (1,1): Patch $\begin{bmatrix}1&0&2\\3&1&2\\1&2&4\end{bmatrix} \rightarrow (1+3+1) - (2+2+4) = 5 - 8 = \mathbf{-3}$
  - (1,2): Patch $\begin{bmatrix}0&2&3\\1&2&0\\2&4&1\end{bmatrix} \rightarrow (0+1+2) - (3+0+1) = 3 - 4 = \mathbf{-1}$
  - (1,3): Patch $\begin{bmatrix}2&3&4\\2&0&1\\4&1&3\end{bmatrix} \rightarrow (2+2+4) - (4+1+3) = 8 - 8 = \mathbf{0}$

- **Row 3:**

  - (2,0): Patch $\begin{bmatrix}1&3&1\\0&1&2\\2&3&0\end{bmatrix} \rightarrow (1+0+2) - (1+2+0) = 3 - 3 = \mathbf{0}$
  - (2,1): Patch $\begin{bmatrix}3&1&2\\1&2&4\\3&0&1\end{bmatrix} \rightarrow (3+1+3) - (2+4+1) = 7 - 7 = \mathbf{0}$
  - (2,2): Patch $\begin{bmatrix}1&2&0\\2&4&1\\0&1&3\end{bmatrix} \rightarrow (1+2+0) - (0+1+3) = 3 - 4 = \mathbf{-1}$
  - (2,3): Patch $\begin{bmatrix}2&0&1\\4&1&3\\1&3&4\end{bmatrix} \rightarrow (2+4+1) - (1+3+4) = 7 - 8 = \mathbf{-1}$

- **Row 4:**
  - (3,0): Patch $\begin{bmatrix}0&1&2\\2&3&0\\1&2&4\end{bmatrix} \rightarrow (0+2+1) - (2+0+4) = 3 - 6 = \mathbf{-3}$
  - (3,1): Patch $\begin{bmatrix}1&2&4\\3&0&1\\2&4&0\end{bmatrix} \rightarrow (1+3+2) - (4+1+0) = 6 - 5 = \mathbf{1}$
  - (3,2): Patch $\begin{bmatrix}2&4&1\\0&1&3\\4&0&2\end{bmatrix} \rightarrow (2+0+4) - (1+3+2) = 6 - 6 = \mathbf{0}$
  - (3,3): Patch $\begin{bmatrix}4&1&3\\1&3&4\\0&2&1\end{bmatrix} \rightarrow (4+1+0) - (3+4+1) = 5 - 8 = \mathbf{-3}$

**Resulting Feature Map:**

$$
\begin{bmatrix}
2 & 2 & 0 & -3 \\
2 & -3 & -1 & 0 \\
0 & 0 & -1 & -1 \\
-3 & 1 & 0 & -3
\end{bmatrix}
$$

**2) Padding of Size 1**

- **New Input Size:** $6 + 2(1) = 8 \times 8$.
- **New Output Size:** $O = \frac{8 - 3}{1} + 1 = 6$.
- **Influence:** Padding allows the output feature map to maintain the same spatial dimensions ($6 \times 6$) as the original input ($6 \times 6$). It prevents the image from shrinking with every layer and allows the filter to process pixels at the very edge of the image.

---

### **Q3. Fuzzy Defuzzification (10 Marks)**

**Given:**
A fuzzy set $A$ composed of four triangular membership functions:

1.  **Pass:** Centered at 60, Peak $\mu=0.8$. Range approx [50, 70].
2.  **Fair:** Centered at 70, Peak $\mu=0.6$. Range approx [60, 80].
3.  **Good:** Centered at 80, Peak $\mu=0.4$. Range approx [70, 90].
4.  **Very Good:** Centered at 90, Peak $\mu=0.2$. Range approx [80, 100].

_(Note: Assuming symmetric triangles with base width 20 based on the grid)._

#### **(i) Weighted Average Method (5 Marks)**

Formula: $x^* = \frac{\sum \mu(x_i) \cdot x_i}{\sum \mu(x_i)}$
We use the peak values of the membership functions.

- $x_1 = 60, \mu_1 = 0.8$
- $x_2 = 70, \mu_2 = 0.6$
- $x_3 = 80, \mu_3 = 0.4$
- $x_4 = 90, \mu_4 = 0.2$

$$x^* = \frac{(0.8 \times 60) + (0.6 \times 70) + (0.4 \times 80) + (0.2 \times 90)}{0.8 + 0.6 + 0.4 + 0.2}$$
$$x^* = \frac{48 + 42 + 32 + 18}{2.0}$$
$$x^* = \frac{140}{2} = \mathbf{70}$$

#### **(ii) Center of Sums Method (5 Marks)**

Formula: $x^* = \frac{\sum A_i \cdot \bar{x}_i}{\sum A_i}$
Where $A_i$ is the area of the $i$-th membership function and $\bar{x}_i$ is its centroid.
Area of triangle = $\frac{1}{2} \times \text{base} \times \text{height}$.
Base for all seems to be $20$ (e.g., 50 to 70).

- **Pass:** Area $A_1 = 0.5 \times 20 \times 0.8 = 8$. Centroid $\bar{x}_1 = 60$.
- **Fair:** Area $A_2 = 0.5 \times 20 \times 0.6 = 6$. Centroid $\bar{x}_2 = 70$.
- **Good:** Area $A_3 = 0.5 \times 20 \times 0.4 = 4$. Centroid $\bar{x}_3 = 80$.
- **VG:** Area $A_4 = 0.5 \times 20 \times 0.2 = 2$. Centroid $\bar{x}_4 = 90$.

$$x^* = \frac{(8 \times 60) + (6 \times 70) + (4 \times 80) + (2 \times 90)}{8 + 6 + 4 + 2}$$
$$x^* = \frac{480 + 420 + 320 + 180}{20}$$
$$x^* = \frac{1400}{20} = \mathbf{70}$$

---

### **Q4. Fuzzy Set Operations (10 Marks)**

**Given:**
$U = \{a10, b52, c130, f2, f9\}$
$A = \{ \frac{0.3}{a10} + \frac{0.4}{b52} + \frac{0.2}{c130} + \frac{0.1}{f2} + \frac{1}{f9} \}$
$B = \{ \frac{0.1}{a10} + \frac{0.2}{b52} + \frac{0.8}{c130} + \frac{0.7}{f2} + \frac{0}{f9} \}$

**(a) $A \cup B$ (Union - Max)**
$\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))$
$= \{ \frac{0.3}{a10} + \frac{0.4}{b52} + \frac{0.8}{c130} + \frac{0.7}{f2} + \frac{1}{f9} \}$

**(b) $A \cap B$ (Intersection - Min)**
$\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))$
$= \{ \frac{0.1}{a10} + \frac{0.2}{b52} + \frac{0.2}{c130} + \frac{0.1}{f2} + \frac{0}{f9} \}$

**(c) $\overline{A}$ (Complement)**
$\mu_{\overline{A}}(x) = 1 - \mu_A(x)$
$= \{ \frac{0.7}{a10} + \frac{0.6}{b52} + \frac{0.8}{c130} + \frac{0.9}{f2} + \frac{0}{f9} \}$

**(d) $\overline{B}$ (Complement)**
$\mu_{\overline{B}}(x) = 1 - \mu_B(x)$
$= \{ \frac{0.9}{a10} + \frac{0.8}{b52} + \frac{0.2}{c130} + \frac{0.3}{f2} + \frac{1}{f9} \}$

**(e) $A | B$ (Difference $A - B = A \cap \overline{B}$)**
$\mu_{A|B}(x) = \min(\mu_A(x), 1 - \mu_B(x))$

- a10: min(0.3, 0.9) = 0.3
- b52: min(0.4, 0.8) = 0.4
- c130: min(0.2, 0.2) = 0.2
- f2: min(0.1, 0.3) = 0.1
- f9: min(1, 1) = 1
  $= \{ \frac{0.3}{a10} + \frac{0.4}{b52} + \frac{0.2}{c130} + \frac{0.1}{f2} + \frac{1}{f9} \}$

**(f) $B | A$ (Difference $B - A = B \cap \overline{A}$)**
$\mu_{B|A}(x) = \min(\mu_B(x), 1 - \mu_A(x))$

- a10: min(0.1, 0.7) = 0.1
- b52: min(0.2, 0.6) = 0.2
- c130: min(0.8, 0.8) = 0.8
- f2: min(0.7, 0.9) = 0.7
- f9: min(0, 0) = 0
  $= \{ \frac{0.1}{a10} + \frac{0.2}{b52} + \frac{0.8}{c130} + \frac{0.7}{f2} + \frac{0}{f9} \}$

**(g) $\overline{A \cup B}$**
Same as $\overline{A} \cap \overline{B}$ (De Morgan's Law).
Complement of result (a).
$= \{ \frac{0.7}{a10} + \frac{0.6}{b52} + \frac{0.2}{c130} + \frac{0.3}{f2} + \frac{0}{f9} \}$

**(h) $\overline{A \cap B}$**
Same as $\overline{A} \cup \overline{B}$.
Complement of result (b).
$= \{ \frac{0.9}{a10} + \frac{0.8}{b52} + \frac{0.8}{c130} + \frac{0.9}{f2} + \frac{1}{f9} \}$

**(i) $\overline{A} \cup \overline{B}$**
Same as (h).
$= \{ \frac{0.9}{a10} + \frac{0.8}{b52} + \frac{0.8}{c130} + \frac{0.9}{f2} + \frac{1}{f9} \}$

**(j) $\overline{B} \cup A$**
Union of (d) and A.

- a10: max(0.9, 0.3) = 0.9
- b52: max(0.8, 0.4) = 0.8
- c130: max(0.2, 0.2) = 0.2
- f2: max(0.3, 0.1) = 0.3
- f9: max(1, 1) = 1
  $= \{ \frac{0.9}{a10} + \frac{0.8}{b52} + \frac{0.2}{c130} + \frac{0.3}{f2} + \frac{1}{f9} \}$


# Soft Computing End Semester Exam Solutions (Part 2)

**Subject:** Principles of Soft Computing (CSE 412)
**Date:** Dec 2024 Paper Solution

---

## Section A (Continued)

### **Q5. Genetic Algorithm (10 Marks)**

#### **a) Concept of Genetic Algorithm (3 Marks)**

**Concept:**
A Genetic Algorithm (GA) is a search heuristic inspired by Charles Darwin's theory of natural selection. It reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

**Replication of Natural Selection:**

1.  **Selection:** Individuals are selected from the population to be parents based on their "fitness" (how well they solve the problem). This mimics "Survival of the Fittest".
2.  **Crossover (Reproduction):** Genes from two parents are combined to create new offspring, mimicking biological reproduction.
3.  **Mutation:** Random changes are introduced to the genes to maintain diversity, mimicking biological mutation.

**Labeled Diagram:**
_(You should draw a flowchart like this)_

```
[Start] -> [Initial Population] -> [Fitness Calculation]
                                         |
                                         v
       [Termination Criteria Met?] -> (Yes) -> [End/Best Solution]
                 | (No)
                 v
            [Selection]
                 |
                 v
            [Crossover]
                 |
                 v
             [Mutation]
                 |
                 v
        [New Population] -> (Loop back to Fitness Calculation)
```

#### **b) Numerical Problem (7 Marks)**

**Given:**

- Function: $f(x) = x^2 + x + 1$
- Range: $[1, 63]$ (Requires 6 bits, as $2^6 = 64$)
- Initial Population: 20, 35, 54, 33, 60, 12
- Operations: Binary Encoding, Roulette-wheel selection, Single-point crossover (4th position), Mutation mask 000001.

**Step 1: Fitness Calculation**
Calculate $f(x)$ for each individual.

| Individual | Binary (6-bit) | Calculation ($x^2 + x + 1$) | Fitness $f(x)$ |
| :--------- | :------------- | :-------------------------- | :------------- |
| 20         | 010100         | $400 + 20 + 1$              | 421            |
| 35         | 100011         | $1225 + 35 + 1$             | 1261           |
| 54         | 110110         | $2916 + 54 + 1$             | 2971           |
| 33         | 100001         | $1089 + 33 + 1$             | 1123           |
| 60         | 111100         | $3600 + 60 + 1$             | 3661           |
| 12         | 001100         | $144 + 12 + 1$              | 157            |
| **Total**  |                |                             | **9594**       |

**Step 2: Selection (Roulette Wheel)**
We calculate the probability of selection $P_i = \frac{f_i}{\sum f}$.

- P(20) = 0.04
- P(35) = 0.13
- P(54) = 0.31
- P(33) = 0.12
- P(60) = 0.38
- P(12) = 0.02

_Note: In a real exam without random numbers provided, we typically proceed by pairing the individuals as they appear in the list or assuming the fittest are selected. Here, we will apply the operations to the **given pairs** (20, 35), (54, 33), and (60, 12) to demonstrate the mechanism._

**Step 3: Crossover (Single-point at 4th position)**
Split after the 4th bit from the left.
Mask: `XXXX | XX`

**Pair 1: 20 & 35**

- 20: `0101` | `00`
- 35: `1000` | `11`
- Offspring 1: `0101` + `11` = `010111` (23)
- Offspring 2: `1000` + `00` = `100000` (32)

**Pair 2: 54 & 33**

- 54: `1101` | `10`
- 33: `1000` | `01`
- Offspring 3: `1101` + `01` = `110101` (53)
- Offspring 4: `1000` + `10` = `100010` (34)

**Pair 3: 60 & 12**

- 60: `1111` | `00`
- 12: `0011` | `00`
- Offspring 5: `1111` + `00` = `111100` (60)
- Offspring 6: `0011` + `00` = `001100` (12)

**Step 4: Mutation (Mask 000001)**
Apply XOR with `000001` (Flip the last bit).

- Off1 (23): `010111` -> `010110` (**22**)
- Off2 (32): `100000` -> `100001` (**33**)
- Off3 (53): `110101` -> `110100` (**52**)
- Off4 (34): `100010` -> `100011` (**35**)
- Off5 (60): `111100` -> `111101` (**61**)
- Off6 (12): `001100` -> `001101` (**13**)

**Step 5: Calculate Maximum Fitness**
The new population is {22, 33, 52, 35, 61, 13}.
The maximum value is **61**.

Calculate fitness for 61:
$f(61) = 61^2 + 61 + 1$
$f(61) = 3721 + 61 + 1 = \mathbf{3783}$

**Answer:** The maximum fitness value after one iteration is **3783**.

---

### **Q6. Genetic Algorithm Techniques (10 Marks)**

#### **(i) Tournament Selection (5 Marks)**

**Definition:**
Tournament selection is a method of selecting individuals from the population for reproduction. It involves running several "tournaments" among a few individuals chosen at random from the population.

**Process:**

1.  Select $k$ individuals randomly from the population ($k$ is the tournament size).
2.  Compare the fitness of these $k$ individuals.
3.  The individual with the highest fitness wins the tournament and is selected as a parent.
4.  Repeat the process to select more parents.

**Advantages:**

- Efficient to implement.
- Works well with parallel architectures.
- Selection pressure can be adjusted by changing the tournament size $k$ (larger $k$ = higher pressure).

#### **(ii) Mutation Techniques (5 Marks)**

Mutation introduces random changes to offspring to maintain genetic diversity and prevent premature convergence.

**1. Bit Flip Mutation (Binary Encoding):**

- Inverts a bit (0 to 1 or 1 to 0) at a random position.
- _Example:_ `101001` -> Mutate 3rd bit -> `100001`.

**2. Swap Mutation (Permutation Encoding):**

- Select two positions at random and swap their values.
- _Example:_ `[1 2 3 4 5]` -> Swap pos 2 and 4 -> `[1 4 3 2 5]`.

**3. Scramble Mutation:**

- Choose a subset of genes and scramble (shuffle) their order randomly.
- _Example:_ `[1 2 3 4 5]` -> Scramble (2,3,4) -> `[1 4 2 3 5]`.

**4. Inversion Mutation:**

- Select a subset of genes and reverse their order.
- _Example:_ `[1 2 3 4 5]` -> Invert (2 to 4) -> `[1 4 3 2 5]`.

---

### **Q7. Neuro-Fuzzy Hybrid System (10 Marks)**

#### **a) Architecture of a Typical Neuro-Fuzzy System (5 Marks)**

A Neuro-Fuzzy system (like ANFIS - Adaptive Neuro-Fuzzy Inference System) combines the learning capability of Neural Networks with the reasoning capability of Fuzzy Logic.

**Architecture Layers (5-Layer Structure):**

1.  **Layer 1 (Input Layer):** Passes input values (crisp) to the next layer. No computation.
2.  **Layer 2 (Fuzzification Layer):** Each node represents a fuzzy set (e.g., Low, High). It calculates membership degrees using membership functions (e.g., Gaussian, Bell).
3.  **Layer 3 (Rule Layer):** Each node represents a fuzzy rule (e.g., IF A is Low AND B is High). It performs the "AND" operation (firing strength calculation).
4.  **Layer 4 (Normalization Layer):** Calculates the normalized firing strength of each rule (ratio of rule strength to sum of all rule strengths).
5.  **Layer 5 (Defuzzification/Output Layer):** Computes the final output by summing the weighted outputs of the rules.

_(You should draw a diagram showing inputs connecting to membership functions, then to rule nodes, then to normalization, and finally to a single output)._

#### **b) Learning/Adaptation vs. Uncertainty/Reasoning (5 Marks)**

**Neural Networks (Learning & Adaptation):**

- **Role:** The neural network component handles the "tuning" of the system.
- **Mechanism:** It uses algorithms like Backpropagation or Gradient Descent to adjust parameters.
- **What it learns:** It learns the shapes of membership functions (e.g., width and center of a Bell curve) and the weights of the rules based on input-output data. This allows the system to adapt to new data without manual intervention.

**Fuzzy Logic (Uncertainty & Reasoning):**

- **Role:** The fuzzy logic component handles the "structure" and "inference" of the system.
- **Mechanism:** It uses linguistic variables (e.g., "Hot", "Cold") and IF-THEN rules.
- **Handling Uncertainty:** It can process vague, noisy, or imprecise information (e.g., "Temperature is somewhat high") which standard neural networks struggle to interpret explicitly. It provides a transparent reasoning framework (White Box model) compared to the Black Box nature of pure Neural Networks.

**Summary:** The hybrid system uses Fuzzy Logic to _represent_ knowledge in a human-readable way and Neural Networks to _optimize_ that knowledge using data.
