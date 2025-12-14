# Soft Computing Assignment - Comprehensive Solutions (Part 1)

**Complete Exam-Style Solutions with Detailed Explanations**

This document provides step-by-step solutions for Questions 1-7 of the assignment. Each solution includes formulas, detailed calculations, and explanations to help you understand the concepts clearly before your exam.

---

## Q1 & Q2: FUZZY LOGIC - SMART HOME LIGHTING SYSTEM

### Question 1: System Description

A smart home lighting system uses fuzzy logic with two fuzzy sets:

- **Set A (Moderate Lighting):** Core = 400 lux, Support = (200, 600)
- **Set B (Bright Lighting):** Core = 500 lux, Support = (300, 700)
- **Universe:** X = [0, 1000] lux

---

### Question 2(a): Sketch Triangular Membership Functions (5 Marks)

**Answer:**

**Understanding Triangular Membership Functions:**

- **Core:** Where membership = 1 (peak of triangle)
- **Support:** Range where membership > 0 (base of triangle)
- **Triangular shape:** Linear increase to peak, then linear decrease

**Fuzzy Set A: Moderate Lighting**

```
Points: (200, 0), (400, 1), (600, 0)

     μ_A(x)
      1.0 |        /\
          |       /  \
      0.5 |      /    \
          |     /      \
      0.0 |____/________\____
          0   200  400  600  1000  (lux)
               ↑    ↑    ↑
            Support Core Support
            starts  peak  ends
```

**Fuzzy Set B: Bright Lighting**

```
Points: (300, 0), (500, 1), (700, 0)

     μ_B(x)
      1.0 |          /\
          |         /  \
      0.5 |        /    \
          |       /      \
      0.0 |______/________\____
          0   300  500  700  1000  (lux)
                 ↑    ↑    ↑
              Support Core Support
```

**Combined Sketch:**

```
     μ(x)
      1.0 |        A    B
          |       /\   /\
      0.5 |      /  \ /  \
          |     /    X    \
      0.0 |____/____/______\____
          0   200 400 500 700  (lux)
                   ↑
                Overlap region
```

**Key Points to Label:**

- Set A: Support (200, 600), Core = 400
- Set B: Support (300, 700), Core = 500
- Overlap region: 300-600 lux

---

### Question 2(b): Compute Membership Values (2 Marks)

**Answer:**

**Formula for Triangular Membership Function:**

📝 **Note:** For a triangular fuzzy set with points (a, 0), (b, 1), (c, 0):

```
μ(x) = {
    0,                if x ≤ a or x ≥ c
    (x - a)/(b - a),  if a < x ≤ b  (rising slope)
    (c - x)/(c - b),  if b < x < c  (falling slope)
}
```

**Why this formula?**

- Rising slope: Linear increase from a to b
- Falling slope: Linear decrease from b to c
- Outside support: Membership is 0

---

**For x = 350 lux:**

**Set A (200, 400, 600):**

```
350 is between 200 and 400 (rising slope)
μ_A(350) = (350 - 200)/(400 - 200)
         = 150/200
         = 0.75
```

**Set B (300, 500, 700):**

```
350 is between 300 and 500 (rising slope)
μ_B(350) = (350 - 300)/(500 - 300)
         = 50/200
         = 0.25
```

**Answer:** μ_A(350) = **0.75**, μ_B(350) = **0.25**

---

**For x = 450 lux:**

**Set A (200, 400, 600):**

```
450 is between 400 and 600 (falling slope)
μ_A(450) = (600 - 450)/(600 - 400)
         = 150/200
         = 0.75
```

**Set B (300, 500, 700):**

```
450 is between 300 and 500 (rising slope)
μ_B(450) = (450 - 300)/(500 - 300)
         = 150/200
         = 0.75
```

**Answer:** μ_A(450) = **0.75**, μ_B(450) = **0.75**

---

**For x = 550 lux:**

**Set A (200, 400, 600):**

```
550 is between 400 and 600 (falling slope)
μ_A(550) = (600 - 550)/(600 - 400)
         = 50/200
         = 0.25
```

**Set B (300, 500, 700):**

```
550 is between 500 and 700 (falling slope)
μ_B(550) = (700 - 550)/(700 - 500)
         = 150/200
         = 0.75
```

**Answer:** μ_A(550) = **0.25**, μ_B(550) = **0.75**

---

**Summary Table:**

| Brightness (lux) | μ_A(x) | μ_B(x) |
| ---------------- | ------ | ------ |
| 350              | 0.75   | 0.25   |
| 450              | 0.75   | 0.75   |
| 550              | 0.25   | 0.75   |

---

### Question 2(c): Fuzzy Intersection and Union (3 Marks)

**Answer:**

**Standard Fuzzy Operators:**

📝 **Note:**

- **Intersection (A ∩ B):** MIN operator → μ_A∩B(x) = min(μ_A(x), μ_B(x))

  - **Why?** Both conditions must be satisfied simultaneously
  - Takes the **more restrictive** (smaller) membership

- **Union (A ∪ B):** MAX operator → μ_A∪B(x) = max(μ_A(x), μ_B(x))
  - **Why?** At least one condition must be satisfied
  - Takes the **less restrictive** (larger) membership

---

**For x = 350 lux:**

**Intersection A ∩ B:**

```
μ_A∩B(350) = min(μ_A(350), μ_B(350))
           = min(0.75, 0.25)
           = 0.25
```

_Interpretation:_ 350 lux has 0.25 degree of being "both moderate AND bright"

**Union A ∪ B:**

```
μ_A∪B(350) = max(μ_A(350), μ_B(350))
           = max(0.75, 0.25)
           = 0.75
```

_Interpretation:_ 350 lux has 0.75 degree of being "moderate OR bright"

---

**For x = 450 lux:**

**Intersection A ∩ B:**

```
μ_A∩B(450) = min(0.75, 0.75)
           = 0.75
```

_Interpretation:_ 450 lux strongly belongs to both sets (optimal overlap)

**Union A ∪ B:**

```
μ_A∪B(450) = max(0.75, 0.75)
           = 0.75
```

_Note:_ Same value because both memberships are equal

---

**For x = 550 lux:**

**Intersection A ∩ B:**

```
μ_A∩B(550) = min(0.25, 0.75)
           = 0.25
```

**Union A ∪ B:**

```
μ_A∪B(550) = max(0.25, 0.75)
           = 0.75
```

---

**Complete Answer Table:**

| x (lux) | μ_A(x) | μ_B(x) | A∩B (min) | A∪B (max) |
| ------- | ------ | ------ | --------- | --------- |
| 350     | 0.75   | 0.25   | **0.25**  | **0.75**  |
| 450     | 0.75   | 0.75   | **0.75**  | **0.75**  |
| 550     | 0.25   | 0.75   | **0.25**  | **0.75**  |

---

## Q2 (CONTINUATION): SMART GREENHOUSE HUMIDITY SYSTEM

### Part (a): Sketch Membership Functions (2 Marks)

**Answer:**

**Fuzzy Set A: Optimal Humidity**

- Support: (35, 75)%
- Core: [45, 65]% (flat top - trapezoidal!)

📝 **Note:** This is a **trapezoidal** membership function, not triangular!

- Core has a range [45, 65] where μ = 1 (flat top)
- This means all values from 45% to 65% are equally "optimal"

```
     μ_A(x)
      1.0 |      ______
          |     /      \
      0.5 |    /        \
          |   /          \
      0.0 |__/____________\__
          0  35  45  65  75  100 (%)
             ↑   ↑____↑   ↑
           Support  Core  Support
```

**Points:** (35, 0), (45, 1), (65, 1), (75, 0)

---

**Fuzzy Set B: High Humidity**

- Support: (60, 95)%
- Core: [70, 85]% (trapezoidal)

```
     μ_B(x)
      1.0 |          ______
          |         /      \
      0.5 |        /        \
          |       /          \
      0.0 |______/____________\__
          0    60  70  85  95  100 (%)
                ↑   ↑____↑   ↑
```

**Points:** (60, 0), (70, 1), (85, 1), (95, 0)

---

### Part (b): Compute Membership Values (3 Marks)

**Answer:**

**Formula for Trapezoidal Membership Function:**

📝 **Note:** For trapezoidal with points (a, 0), (b, 1), (c, 1), (d, 0):

```
μ(x) = {
    0,                if x ≤ a or x ≥ d
    (x - a)/(b - a),  if a < x ≤ b  (rising)
    1,                if b < x ≤ c  (flat top)
    (d - x)/(d - c),  if c < x < d  (falling)
}
```

---

**For x = 55%:**

**Set A (35, 45, 65, 75):**

```
55 is in the core [45, 65]
μ_A(55) = 1
```

_Fully optimal humidity_

**Set B (60, 70, 85, 95):**

```
55 is before 60 (outside support)
μ_B(55) = 0
```

_Not high humidity at all_

**Answer:** μ_A(55) = **1.0**, μ_B(55) = **0.0**

---

**For x = 72%:**

**Set A (35, 45, 65, 75):**

```
72 is between 65 and 75 (falling slope)
μ_A(72) = (75 - 72)/(75 - 65)
        = 3/10
        = 0.3
```

**Set B (60, 70, 85, 95):**

```
72 is in the core [70, 85]
μ_B(72) = 1
```

**Answer:** μ_A(72) = **0.3**, μ_B(72) = **1.0**

---

**For x = 88%:**

**Set A (35, 45, 65, 75):**

```
88 is beyond 75 (outside support)
μ_A(88) = 0
```

**Set B (60, 70, 85, 95):**

```
88 is between 85 and 95 (falling slope)
μ_B(88) = (95 - 88)/(95 - 85)
        = 7/10
        = 0.7
```

**Answer:** μ_A(88) = **0.0**, μ_B(88) = **0.7**

---

**Summary:**

| Humidity (%) | μ_A(x) | μ_B(x) |
| ------------ | ------ | ------ |
| 55           | 1.0    | 0.0    |
| 72           | 0.3    | 1.0    |
| 88           | 0.0    | 0.7    |

---

### Part (c): Fuzzy Operations (2 Marks)

**Answer:**

Using MIN for intersection, MAX for union:

**For x = 55%:**

```
A ∩ B: min(1.0, 0.0) = 0.0
A ∪ B: max(1.0, 0.0) = 1.0
```

**For x = 72%:**

```
A ∩ B: min(0.3, 1.0) = 0.3
A ∪ B: max(0.3, 1.0) = 1.0
```

**For x = 88%:**

```
A ∩ B: min(0.0, 0.7) = 0.0
A ∪ B: max(0.0, 0.7) = 0.7
```

**Complete Table:**

| x (%) | μ_A | μ_B | A∩B | A∪B |
| ----- | --- | --- | --- | --- |
| 55    | 1.0 | 0.0 | 0.0 | 1.0 |
| 72    | 0.3 | 1.0 | 0.3 | 1.0 |
| 88    | 0.0 | 0.7 | 0.0 | 0.7 |

---

## Q3 & Q4: FUZZY SET PROPERTIES - AIR QUALITY INDEX

### Question 3: Given Fuzzy Set

**Fuzzy Set D (Good Air Quality):**

| AQI (x) | 0   | 25  | 50  | 75  | 100 | 125 | 150 |
| ------- | --- | --- | --- | --- | --- | --- | --- |
| μ_D(x)  | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 | 0.1 | 0.0 |

---

### Question 4.1: Support and Crossover Points (2 Marks)

**Answer:**

**Support:**

📝 **Note:** Support = set of all elements where membership > 0

```
Support(D) = {x ∈ X | μ_D(x) > 0}
           = {0, 25, 50, 75, 100, 125}
```

**Why not 150?** Because μ_D(150) = 0 (not greater than 0)

**Answer:** Support = **{0, 25, 50, 75, 100, 125}**

---

**Crossover Points:**

📝 **Note:** Crossover point = element where μ(x) = 0.5

Looking at the table:

- At x = 25: μ = 0.8 (above 0.5)
- At x = 50: μ = 0.6 (above 0.5)
- At x = 75: μ = 0.4 (below 0.5)

**Crossover point is between 50 and 75**

Using linear interpolation:

```
At x = 50: μ = 0.6
At x = 75: μ = 0.4

For μ = 0.5:
(0.5 - 0.6)/(0.4 - 0.6) = (x - 50)/(75 - 50)
(-0.1)/(-0.2) = (x - 50)/25
0.5 = (x - 50)/25
x - 50 = 12.5
x = 62.5
```

**Answer:** Crossover point = **62.5 AQI**

---

### Question 4.2: Intensification and Diffusion (2 Marks)

**Answer:**

**At x = 50 AQI, μ_D(50) = 0.6**

**Intensification (CON - Concentration):**

📝 **Note:** Intensification sharpens the fuzzy set by **squaring** membership values

- **Formula:** CON(D) → μ_CON(x) = [μ_D(x)]²
- **Why?** Values < 1 become smaller when squared (0.6² = 0.36 < 0.6)
- **Effect:** Makes "somewhat good" → "less good", emphasizes extremes

```
μ_excellent(50) = [μ_D(50)]²
                = (0.6)²
                = 0.36
```

**Interpretation:** "Excellent air quality" requires higher standards, so 50 AQI has lower membership.

---

**Diffusion (DIL - Dilation):**

📝 **Note:** Diffusion broadens the fuzzy set by taking **square root**

- **Formula:** DIL(D) → μ_DIL(x) = √[μ_D(x)]
- **Why?** Square root makes values larger (√0.6 = 0.775 > 0.6)
- **Effect:** Makes "somewhat good" → "more good", relaxes criteria

```
μ_fair(50) = √[μ_D(50)]
           = √0.6
           = 0.775
```

**Interpretation:** "Fair air quality" has relaxed standards, so 50 AQI has higher membership.

---

**Answer:**

- **Excellent Air Quality (Intensification):** μ = **0.36**
- **Fair Air Quality (Diffusion):** μ = **0.775**

**Verification:**

```
Original: 0.6
Intensified: 0.36 (smaller ✓)
Diffused: 0.775 (larger ✓)
```

---

### Question 4.3: Cardinality (1 Mark)

**Answer:**

**Cardinality:**

📝 **Note:** Cardinality = sum of all membership values

- **Formula:** |D| = Σ μ_D(x) for all x ∈ X
- **Why?** Measures the "size" or "fuzziness" of the set
- Crisp set: cardinality = number of elements
- Fuzzy set: cardinality = sum of memberships (accounts for partial membership)

```
|D| = μ_D(0) + μ_D(25) + μ_D(50) + μ_D(75) + μ_D(100) + μ_D(125) + μ_D(150)
    = 1.0 + 0.8 + 0.6 + 0.4 + 0.2 + 0.1 + 0.0
    = 3.1
```

**Answer:** Cardinality of D = **3.1**

**Interpretation:** The fuzzy set D has an effective "size" of 3.1 elements (accounting for partial memberships).

---

## Q4: GENETIC ALGORITHM - FITNESS AND CROSSOVER

### Given Information:

**Chromosome:** 8 genes, each digit 0-9
**Fitness function:** f(x) = (a+b) - (c+d) + 2(e+f) - (g+h)

📝 **Note:** This fitness function:

- Rewards high values in positions a, b, e, f (positive terms)
- Penalizes high values in positions c, d, g, h (negative terms)
- Position e and f have double weight (coefficient 2)

**Initial Population:**

```
x₁ = 7 4 3 2 5 6 2 1
x₂ = 9 6 2 3 8 4 1 2
x₃ = 3 2 8 5 4 3 6 4
x₄ = 5 3 6 4 7 2 3 5
```

---

### Part (a): Evaluate Fitness and Rank (3 Marks)

**Answer:**

**Individual x₁ = 7 4 3 2 5 6 2 1:**

```
f(x₁) = (7+4) - (3+2) + 2(5+6) - (2+1)
      = 11 - 5 + 2(11) - 3
      = 11 - 5 + 22 - 3
      = 25
```

**Individual x₂ = 9 6 2 3 8 4 1 2:**

```
f(x₂) = (9+6) - (2+3) + 2(8+4) - (1+2)
      = 15 - 5 + 2(12) - 3
      = 15 - 5 + 24 - 3
      = 31
```

**Individual x₃ = 3 2 8 5 4 3 6 4:**

```
f(x₃) = (3+2) - (8+5) + 2(4+3) - (6+4)
      = 5 - 13 + 2(7) - 10
      = 5 - 13 + 14 - 10
      = -4
```

**Individual x₄ = 5 3 6 4 7 2 3 5:**

```
f(x₄) = (5+3) - (6+4) + 2(7+2) - (3+5)
      = 8 - 10 + 2(9) - 8
      = 8 - 10 + 18 - 8
      = 8
```

---

**Ranked Population (Fittest to Least Fit):**

| Rank | Individual | Chromosome      | Fitness |
| ---- | ---------- | --------------- | ------- |
| 1st  | x₂         | 9 6 2 3 8 4 1 2 | **31**  |
| 2nd  | x₁         | 7 4 3 2 5 6 2 1 | **25**  |
| 3rd  | x₄         | 5 3 6 4 7 2 3 5 | **8**   |
| 4th  | x₃         | 3 2 8 5 4 3 6 4 | **-4**  |

---

### Part (b.i): Single-Point Crossover at Middle (1 Mark)

**Answer:**

**Parents:** x₂ (fittest) and x₁ (2nd fittest)

📝 **Note:** Single-point crossover at middle point

- Chromosome length = 8 genes
- Middle point = after position 4 (between gene 4 and 5)
- **How it works:** Split both parents at the cut point, swap tails

```
Parent x₂: 9 6 2 3 | 8 4 1 2
Parent x₁: 7 4 3 2 | 5 6 2 1
           --------+--------
              ↓ cut point after position 4

Offspring O₁: 9 6 2 3 | 5 6 2 1  (head from x₂, tail from x₁)
Offspring O₂: 7 4 3 2 | 8 4 1 2  (head from x₁, tail from x₂)
```

**Answer:**

- **O₁ = 9 6 2 3 5 6 2 1**
- **O₂ = 7 4 3 2 8 4 1 2**

---

### Part (b.ii): Two-Point Crossover (2 Marks)

**Answer:**

**Parents:** x₄ (3rd) and x₃ (4th)

📝 **Note:** Two-point crossover at positions b and f

- Position b = after gene 2
- Position f = after gene 6
- **How it works:** Keep outer segments from one parent, middle segment from other

```
Position:     1 2 3 4 5 6 7 8
Parent x₄:    5 3 | 6 4 7 2 | 3 5
Parent x₃:    3 2 | 8 5 4 3 | 6 4
              ----+---------+----
                  ↑         ↑
               Point b    Point f

Offspring O₃: 5 3 | 8 5 4 3 | 3 5  (outer from x₄, middle from x₃)
Offspring O₄: 3 2 | 6 4 7 2 | 6 4  (outer from x₃, middle from x₄)
```

**Answer:**

- **O₃ = 5 3 8 5 4 3 3 5**
- **O₄ = 3 2 6 4 7 2 6 4**

---

### Part (b.iii): Modified Population (1 Mark)

**Answer:**

**New Population after Crossover:**

```
O₁ = 9 6 2 3 5 6 2 1  (from single-point crossover)
O₂ = 7 4 3 2 8 4 1 2  (from single-point crossover)
O₃ = 5 3 8 5 4 3 3 5  (from two-point crossover)
O₄ = 3 2 6 4 7 2 6 4  (from two-point crossover)
```

---

### Part (b.iv): Final Fitness Values (3 Marks)

**Answer:**

**Offspring O₁ = 9 6 2 3 5 6 2 1:**

```
f(O₁) = (9+6) - (2+3) + 2(5+6) - (2+1)
      = 15 - 5 + 2(11) - 3
      = 15 - 5 + 22 - 3
      = 29
```

**Offspring O₂ = 7 4 3 2 8 4 1 2:**

```
f(O₂) = (7+4) - (3+2) + 2(8+4) - (1+2)
      = 11 - 5 + 2(12) - 3
      = 11 - 5 + 24 - 3
      = 27
```

**Offspring O₃ = 5 3 8 5 4 3 3 5:**

```
f(O₃) = (5+3) - (8+5) + 2(4+3) - (3+5)
      = 8 - 13 + 2(7) - 8
      = 8 - 13 + 14 - 8
      = 1
```

**Offspring O₄ = 3 2 6 4 7 2 6 4:**

```
f(O₄) = (3+2) - (6+4) + 2(7+2) - (6+4)
      = 5 - 10 + 2(9) - 10
      = 5 - 10 + 18 - 10
      = 3
```

---

**Final Population Summary:**

| Offspring | Chromosome      | Fitness | Improvement         |
| --------- | --------------- | ------- | ------------------- |
| O₁        | 9 6 2 3 5 6 2 1 | **29**  | Better than x₁ (25) |
| O₂        | 7 4 3 2 8 4 1 2 | **27**  | Better than x₁ (25) |
| O₃        | 5 3 8 5 4 3 3 5 | **1**   | Better than x₃ (-4) |
| O₄        | 3 2 6 4 7 2 6 4 | **3**   | Better than x₃ (-4) |

**Observation:** All offspring improved compared to their parent pairs! ✓

---

## Q5: ROULETTE WHEEL SELECTION

### Given Information:

**Population fitness:** [f₁, f₂, f₃, f₄] = [12, 7, 3, 8]
**Task:** Compute selection probability and expected count for 10 parent selections

---

### Solution:

**Step 1: Calculate Total Fitness**

📝 **Note:** Total fitness is the sum of all individual fitnesses

```
Total = Σf_i = 12 + 7 + 3 + 8 = 30
```

---

**Step 2: Calculate Selection Probability**

📝 **Note:** Selection probability = Individual fitness / Total fitness

- **Why?** Probability proportional to fitness (fitter → higher chance)

```
P₁ = f₁/Total = 12/30 = 0.40 = 40%
P₂ = f₂/Total = 7/30  = 0.233 = 23.3%
P₃ = f₃/Total = 3/30  = 0.10 = 10%
P₄ = f₄/Total = 8/30  = 0.267 = 26.7%
```

**Verification:** 0.40 + 0.233 + 0.10 + 0.267 = 1.00 ✓

---

**Step 3: Calculate Expected Count**

📝 **Note:** Expected count = Probability × Number of selections

- **Formula:** E_i = P_i × N
- **Interpretation:** On average, how many times each chromosome will be selected

For N = 10 selections:

```
E₁ = 0.40 × 10 = 4.0 times
E₂ = 0.233 × 10 = 2.33 times
E₃ = 0.10 × 10 = 1.0 time
E₄ = 0.267 × 10 = 2.67 times
```

**Verification:** 4.0 + 2.33 + 1.0 + 2.67 = 10.0 ✓

---

**Complete Answer Table:**

| Chromosome | Fitness | Probability   | Expected Count (N=10) |
| ---------- | ------- | ------------- | --------------------- |
| C₁         | 12      | 0.400 (40%)   | **4.0**               |
| C₂         | 7       | 0.233 (23.3%) | **2.33**              |
| C₃         | 3       | 0.100 (10%)   | **1.0**               |
| C₄         | 8       | 0.267 (26.7%) | **2.67**              |
| **Total**  | **30**  | **1.000**     | **10.0**              |

---

**Interpretation:**

- C₁ (highest fitness) will be selected ~4 times on average
- C₃ (lowest fitness) will be selected ~1 time on average
- Fitter individuals get more reproduction opportunities (survival of the fittest!)

---

## Q6: GENETIC ALGORITHM - SELECTION AND OPERATORS

### Part (a.1): Roulette Wheel Selection (3 Marks)

**Given Population:**

| Individual | 1   | 2   | 3   | 4   | 5   | 6   |
| ---------- | --- | --- | --- | --- | --- | --- |
| Fitness    | 3.2 | 4.8 | 2.1 | 5.6 | 3.9 | 4.3 |

---

**Solution:**

**Step 1: Total Fitness**

```
Total = 3.2 + 4.8 + 2.1 + 5.6 + 3.9 + 4.3 = 23.9
```

**Step 2: Selection Probabilities**

```
P₁ = 3.2/23.9 = 0.134 (13.4%)
P₂ = 4.8/23.9 = 0.201 (20.1%)
P₃ = 2.1/23.9 = 0.088 (8.8%)
P₄ = 5.6/23.9 = 0.234 (23.4%)
P₅ = 3.9/23.9 = 0.163 (16.3%)
P₆ = 4.3/23.9 = 0.180 (18.0%)
```

**Step 3: Expected Counts** (assuming 6 selections for new generation)

```
E₁ = 0.134 × 6 = 0.80
E₂ = 0.201 × 6 = 1.21
E₃ = 0.088 × 6 = 0.53
E₄ = 0.234 × 6 = 1.40
E₅ = 0.163 × 6 = 0.98
E₆ = 0.180 × 6 = 1.08
```

---

**Answer Table:**

| Individual | Fitness | Probability | Expected Count |
| ---------- | ------- | ----------- | -------------- |
| 1          | 3.2     | 0.134       | 0.80           |
| 2          | 4.8     | 0.201       | 1.21           |
| 3          | 2.1     | 0.088       | 0.53           |
| **4**      | **5.6** | **0.234**   | **1.40** ✓     |
| 5          | 3.9     | 0.163       | 0.98           |
| 6          | 4.3     | 0.180       | 1.08           |

**Most Likely Parents:**

- **Individual 4** (highest probability 23.4%, expected 1.40 times)
- **Individual 2** (second highest 20.1%, expected 1.21 times)
- **Individual 6** (third highest 18.0%, expected 1.08 times)

---

### Part (a.2): Tournament Selection (2 Marks)

**Given Tournaments:** (1,4), (2,5), (3,6), (4,2), (5,3), (6,1)
**Tournament size:** k = 2
**Selection probability for fitter:** p = 0.8

📝 **Note:** In tournament selection with probability:

- Fitter individual wins with probability p = 0.8
- Less fit individual wins with probability (1-p) = 0.2
- This adds controlled randomness

---

**Solution:**

**Tournament 1: (1,4)**

```
Individual 1: fitness = 3.2
Individual 4: fitness = 5.6 (fitter)

Winner: Individual 4 with probability 0.8
```

**Deterministic winner:** Individual **4**

---

**Tournament 2: (2,5)**

```
Individual 2: fitness = 4.8 (fitter)
Individual 5: fitness = 3.9

Winner: Individual 2 with probability 0.8
```

**Deterministic winner:** Individual **2**

---

**Tournament 3: (3,6)**

```
Individual 3: fitness = 2.1
Individual 6: fitness = 4.3 (fitter)

Winner: Individual 6 with probability 0.8
```

**Deterministic winner:** Individual **6**

---

**Tournament 4: (4,2)**

```
Individual 4: fitness = 5.6 (fitter)
Individual 2: fitness = 4.8

Winner: Individual 4 with probability 0.8
```

**Deterministic winner:** Individual **4**

---

**Tournament 5: (5,3)**

```
Individual 5: fitness = 3.9 (fitter)
Individual 3: fitness = 2.1

Winner: Individual 5 with probability 0.8
```

**Deterministic winner:** Individual **5**

---

**Tournament 6: (6,1)**

```
Individual 6: fitness = 4.3 (fitter)
Individual 1: fitness = 3.2

Winner: Individual 6 with probability 0.8
```

**Deterministic winner:** Individual **6**

---

**Answer:**

**Winners proceeding to next generation:**

- Tournament 1: **Individual 4**
- Tournament 2: **Individual 2**
- Tournament 3: **Individual 6**
- Tournament 4: **Individual 4** (selected again)
- Tournament 5: **Individual 5**
- Tournament 6: **Individual 6** (selected again)

**Selection Count:**

- Individual 2: 1 time
- Individual 4: 2 times (highest fitness, selected twice!)
- Individual 5: 1 time
- Individual 6: 2 times

**Not selected:** Individuals 1 and 3 (lowest fitness values)

---

### Part (b.1): Crossover Operations (2 Marks)

**Given:**

- Fitness function: f(x) = x²
- Range: x ∈ {0, 1, 2, ..., 15}
- Encoding: 4-bit binary

**Initial Population:**

```
P₁ = 0110 → x = 6  → f(6) = 36
P₂ = 1001 → x = 9  → f(9) = 81
P₃ = 1101 → x = 13 → f(13) = 169
P₄ = 0011 → x = 3  → f(3) = 9
```

**Parent Pairs:** (P₁, P₂) and (P₃, P₄)

---

**METHOD 1: Single-Point Crossover at 2nd Bit**

📝 **Note:** Cut after 2nd bit (position 2)

**Pair 1: (P₁, P₂)**

```
P₁: 01|10
P₂: 10|01
    --+--
     ↓ cut after bit 2

Child C₁: 01|01 = 0101
Child C₂: 10|10 = 1010
```

**Decode and compute fitness:**

```
C₁ = 0101 → x = 5  → f(5) = 25
C₂ = 1010 → x = 10 → f(10) = 100
```

---

**Pair 2: (P₃, P₄)**

```
P₃: 11|01
P₄: 00|11
    --+--

Child C₃: 11|11 = 1111
Child C₄: 00|01 = 0001
```

**Decode and compute fitness:**

```
C₃ = 1111 → x = 15 → f(15) = 225
C₄ = 0001 → x = 1  → f(1) = 1
```

---

**Single-Point Crossover Results:**

| Offspring | Binary | Decimal x | f(x) = x² |
| --------- | ------ | --------- | --------- |
| C₁        | 0101   | 5         | 25        |
| C₂        | 1010   | 10        | 100       |
| C₃        | 1111   | 15        | **225** ✓ |
| C₄        | 0001   | 1         | 1         |

---

**METHOD 2: Two-Point Crossover at Bits 1 and 3**

📝 **Note:** Cut after bit 1 and after bit 3

- Keep outer bits from one parent, middle bits from other

**Pair 1: (P₁, P₂)**

```
Position: 1 2 3 4
P₁:       0|1 1|0
P₂:       1|0 0|1
          -+---+-
           ↑   ↑
          cut cut

Child C₁': 0|0 0|0 = 0000 (outer from P₁, middle from P₂)
Child C₂': 1|1 1|1 = 1111 (outer from P₂, middle from P₁)
```

**Decode and compute fitness:**

```
C₁' = 0000 → x = 0  → f(0) = 0
C₂' = 1111 → x = 15 → f(15) = 225
```

---

**Pair 2: (P₃, P₄)**

```
Position: 1 2 3 4
P₃:       1|1 0|1
P₄:       0|0 1|1
          -+---+-

Child C₃': 1|0 1|1 = 1011
Child C₄': 0|1 0|1 = 0101
```

**Decode and compute fitness:**

```
C₃' = 1011 → x = 11 → f(11) = 121
C₄' = 0101 → x = 5  → f(5) = 25
```

---

**Two-Point Crossover Results:**

| Offspring | Binary | Decimal x | f(x) = x² |
| --------- | ------ | --------- | --------- |
| C₁'       | 0000   | 0         | 0         |
| C₂'       | 1111   | 15        | **225** ✓ |
| C₃'       | 1011   | 11        | 121       |
| C₄'       | 0101   | 5         | 25        |

---

### Part (b.2): Mutation (2 Marks)

**Answer:**

**Least fit offspring from two-point crossover:**

- C₁' has fitness = 0 (lowest)

**Original:** C₁' = 0000

📝 **Note:** LSB (Least Significant Bit) = rightmost bit (position 4)

- Bit flip: 0 → 1 or 1 → 0

**Mutation:** Flip LSB (bit 4)

```
Before: 0000
After:  0001  (flipped last bit from 0 to 1)
```

**New value:**

```
Mutated = 0001 → x = 1 → f(1) = 1² = 1
```

**Answer:**

- **Mutated chromosome:** 0001
- **New x:** 1
- **New f(x):** 1

**Improvement:** Fitness increased from 0 to 1 ✓

---

### Part (b.3): Comment on Crossover and Mutation (1 Mark)

**Answer:**

📝 **How GA operators guide toward higher fitness:**

**Crossover:**

- **Combines good traits** from different parents
- Example: C₃ (1111, f=225) emerged by combining high-order bits from P₃ and P₄
- **Exploitation:** Uses existing good genetic material
- Produces offspring that can be **better than both parents** (e.g., C₃ with f=225 > P₃'s 169)

**Mutation:**

- **Introduces new genetic variation** not present in parents
- Prevents population from getting stuck (premature convergence)
- **Exploration:** Searches new regions of solution space
- Small improvements add up (0 → 1 in our example)

**Together:** Crossover exploits known good solutions, mutation explores new possibilities, leading the population toward optimal fitness (max x² = 15² = 225).

---

## Q7: FUZZY LOGIC CONTROLLER - SMART AIR COOLING

### Given Information:

**Fuzzy Sets (Triangular):**

- Low: μ_L(x; 16, 20, 24)
- Medium: μ_M(x; 19, 25, 31)
- High: μ_H(x; 23, 28, 33)

**Output Fan Speeds:**

- s_L = 25%
- s_M = 55%
- s_H = 85%

---

### Part (a): Fuzzy Rule Base (2 Marks)

**Answer:**

📝 **Note:** Rule base maps input conditions to output actions

- Format: IF (Temperature is X) THEN (Fan Speed is Y)

**Fuzzy Rule Base:**

```
Rule 1: IF Temperature is Low THEN Fan Speed is Low (25%)
Rule 2: IF Temperature is Medium THEN Fan Speed is Medium (55%)
Rule 3: IF Temperature is High THEN Fan Speed is High (85%)
```

**Linguistic Interpretation:**

- **Low temperature (16-24°C):** Minimal cooling needed → 25% fan speed
- **Medium temperature (19-31°C):** Moderate cooling → 55% fan speed
- **High temperature (23-33°C):** Maximum cooling → 85% fan speed

---

### Part (b): Compute Memberships and Defuzzified Output (4 Marks)

**Given:** x = 27°C

**Step 1: Compute Membership Values**

📝 **Note:** Use triangular membership formula from Q2

**μ_L(27) for Low (16, 20, 24):**

```
27 > 24 (outside support)
μ_L(27) = 0
```

**μ_M(27) for Medium (19, 25, 31):**

```
27 is between 25 and 31 (falling slope)
μ_M(27) = (31 - 27)/(31 - 25)
        = 4/6
        = 0.667
```

**μ_H(27) for High (23, 28, 33):**

```
27 is between 23 and 28 (rising slope)
μ_H(27) = (27 - 23)/(28 - 23)
        = 4/5
        = 0.8
```

**Membership Values:**

- μ_L(27) = 0
- μ_M(27) = 0.667
- μ_H(27) = 0.8

---

**Step 2: Defuzzification (Centroid Method)**

📝 **Note:** Centroid (Center of Gravity) formula:

```
Output = Σ(μᵢ × sᵢ) / Σμᵢ
```

**Why?** Weighted average of all active rules by their firing strengths

```
Fan Speed = (μ_L × s_L + μ_M × s_M + μ_H × s_H) / (μ_L + μ_M + μ_H)

          = (0 × 25 + 0.667 × 55 + 0.8 × 85) / (0 + 0.667 + 0.8)

          = (0 + 36.685 + 68) / 1.467

          = 104.685 / 1.467

          = 71.36%
```

**Answer:**

- μ_L(27) = 0, μ_M(27) = 0.667, μ_H(27) = 0.8
- **Defuzzified Fan Speed = 71.36%**

---

### Part (c): GA-Tuned System (3 Marks)

**Answer:**

**GA updates Medium to (20, 24, 29)**

**Recompute μ_M(27) for new Medium (20, 24, 29):**

```
27 is between 24 and 29 (falling slope)
μ_M(27) = (29 - 27)/(29 - 24)
        = 2/5
        = 0.4
```

**Updated memberships:**

- μ_L(27) = 0 (unchanged)
- μ_M(27) = 0.4 (changed from 0.667)
- μ_H(27) = 0.8 (unchanged)

---

**New Defuzzified Output:**

```
Fan Speed = (0 × 25 + 0.4 × 55 + 0.8 × 85) / (0 + 0.4 + 0.8)

          = (0 + 22 + 68) / 1.2

          = 90 / 1.2

          = 75%
```

**Answer:**

- New μ_M(27) = 0.4
- **New Fan Speed = 75%**

---

### Part (d): Comparison (1 Mark)

**Answer:**

**Before GA tuning:** 71.36%
**After GA tuning:** 75%

**Difference:** +3.64% increase

**Analysis:**

- GA tuning **increased** the fan speed at 27°C
- This shifts more weight toward "High" temperature category
- **More aggressive cooling** at 27°C

**Which is smoother?**

- **Before (71.36%):** More balanced between Medium and High
- **After (75%):** More decisive High response

**Answer:** The **original configuration (71.36%)** achieves **smoother control** because:

1. It has more overlap between Medium and High at 27°C (μ_M=0.667 vs 0.4)
2. More gradual transition between temperature zones
3. The GA-tuned version responds more sharply, which may cause abrupt changes

However, the GA-tuned version may be **more appropriate** if rapid cooling is needed at borderline temperatures.

---

---

# PART 2: ADVANCED TOPICS

## Q8: GA-FUZZY HYBRID - ADAPTIVE TEMPERATURE CONTROL

### Given Information:

**GA Chromosome encodes:**

```
[a_L, b_L, c_L, a_M, b_M, c_M, a_H, b_H, c_H, s_L, s_M, s_H]
```

**Evolved chromosome:**

```
[16, 20, 24, 20, 24, 29, 23, 28, 33, 25, 55, 85]
```

**Decoded:**

- Low: μ_L(x; 16, 20, 24), output = 25%
- Medium: μ_M(x; 20, 24, 29), output = 55%
- High: μ_H(x; 23, 28, 33), output = 85%

---

### Part (a): Frame Fuzzy Rule Base (2 Marks)

**Answer:**

📝 **Note:** The chromosome directly encodes the fuzzy system configuration

- First 9 values: membership function parameters (3 per fuzzy set)
- Last 3 values: output fan speeds

**Fuzzy Rule Base:**

```
Rule 1: IF Temperature is Low (16-20-24) THEN Fan Speed is 25%
Rule 2: IF Temperature is Medium (20-24-29) THEN Fan Speed is 55%
Rule 3: IF Temperature is High (23-28-33) THEN Fan Speed is 85%
```

**System Description:**

- **Low temperature range:** 16°C to 24°C, peak at 20°C → Minimal cooling (25%)
- **Medium temperature range:** 20°C to 29°C, peak at 24°C → Moderate cooling (55%)
- **High temperature range:** 23°C to 33°C, peak at 28°C → Maximum cooling (85%)

---

### Part (b): Compute Memberships and Defuzzified Output (4 Marks)

**Given:** x = 27°C

**Step 1: Calculate Membership Values**

**μ_L(27) for Low (16, 20, 24):**

```
27 > 24 (outside support)
μ_L(27) = 0
```

**μ_M(27) for Medium (20, 24, 29):**

```
27 is between 24 and 29 (falling slope)

Formula: μ(x) = (c - x)/(c - b)

μ_M(27) = (29 - 27)/(29 - 24)
        = 2/5
        = 0.4
```

**μ_H(27) for High (23, 28, 33):**

```
27 is between 23 and 28 (rising slope)

Formula: μ(x) = (x - a)/(b - a)

μ_H(27) = (27 - 23)/(28 - 23)
        = 4/5
        = 0.8
```

**Membership Summary:**

- μ_L(27) = 0
- μ_M(27) = 0.4
- μ_H(27) = 0.8

---

**Step 2: Defuzzification (Centroid)**

📝 **Note:** Weighted average formula:

```
Output = Σ(μᵢ × sᵢ) / Σμᵢ
```

```
Fan Speed = (μ_L × s_L + μ_M × s_M + μ_H × s_H) / (μ_L + μ_M + μ_H)

          = (0 × 25 + 0.4 × 55 + 0.8 × 85) / (0 + 0.4 + 0.8)

          = (0 + 22 + 68) / 1.2

          = 90 / 1.2

          = 75%
```

**Answer:**

- Memberships: μ_L(27) = 0, μ_M(27) = 0.4, μ_H(27) = 0.8
- **Defuzzified Fan Speed = 75%**

---

### Part (c): GA Re-Tuning (3 Marks)

**Answer:**

**GA updates Medium to (19, 25, 31)**

**Recompute μ_M(27) for new Medium (19, 25, 31):**

```
27 is between 25 and 31 (falling slope)

μ_M(27) = (31 - 27)/(31 - 25)
        = 4/6
        = 0.667
```

**Updated memberships:**

- μ_L(27) = 0 (unchanged)
- μ_M(27) = 0.667 (changed from 0.4)
- μ_H(27) = 0.8 (unchanged)

---

**New Defuzzified Output:**

```
Fan Speed = (0 × 25 + 0.667 × 55 + 0.8 × 85) / (0 + 0.667 + 0.8)

          = (0 + 36.685 + 68) / 1.467

          = 104.685 / 1.467

          = 71.36%
```

**Answer:**

- New μ_M(27) = 0.667
- **New Fan Speed = 71.36%**

---

### Part (d): Configuration Comparison (1 Mark)

**Answer:**

**Before GA tuning (20, 24, 29):** Fan Speed = 75%
**After GA tuning (19, 25, 31):** Fan Speed = 71.36%

**Difference:** -3.64% decrease

**Analysis:**

**Original configuration (75%):**

- More aggressive cooling at 27°C
- Sharper response (lower Medium membership)
- Faster reaction to temperature changes

**GA-tuned configuration (71.36%):**

- Gentler cooling at 27°C
- Smoother response (higher Medium membership)
- More balanced between Medium and High zones

**Which improves adaptability?**

**Answer:** The **GA-tuned configuration (71.36%)** improves adaptability because:

1. **Better overlap:** Higher Medium membership (0.667 vs 0.4) creates smoother transitions
2. **More gradual response:** Prevents sudden fan speed changes that startle occupants
3. **Energy efficiency:** Slightly lower fan speed saves energy while maintaining comfort
4. **Robustness:** Less sensitive to small temperature fluctuations

The GA successfully **optimized for smooth, comfortable control** rather than aggressive cooling.

---

## Q9: GA-BP HYBRID NEURAL NETWORK

### Given Information:

**Network Architecture:**

- Input layer: 2 neurons
- Hidden layer: 2 neurons (ReLU activation)
- Output layer: 1 neuron (Linear activation)

**Data:**

- Input: X = [1, 2]ᵀ
- Target: y = 5
- Learning rate: η = 0.1

**Initial GA Population:**

```
C₁ = [w₁₁, w₁₂, w₂₁, w₂₂, b₁, b₂, v₁, v₂, b_o]
   = [1, 2, 2, 1, 0, 1, 2, 1, 0]

C₂ = [2, 1, 1, 2, 1, 0, 1, 2, 1]
```

---

### Part (a): Compute Network Output and Error (4 Marks)

**Answer:**

📝 **Note:** Network computation flow:

1. Hidden layer: h = ReLU(W₁ · X + b₁)
2. Output layer: ŷ = W₂ · h + b_o
3. Error: E = (1/2)(y - ŷ)²

---

**CHROMOSOME C₁:**

**Step 1: Extract weights and biases**

```
W₁ = [w₁₁  w₁₂] = [1  2]
     [w₂₁  w₂₂]   [2  1]

b₁ = [b₁] = [0]
     [b₂]   [1]

W₂ = [v₁  v₂] = [2  1]

b_o = 0
```

**Step 2: Hidden layer computation**

```
z₁ = w₁₁×x₁ + w₁₂×x₂ + b₁
   = 1×1 + 2×2 + 0
   = 1 + 4 + 0
   = 5

z₂ = w₂₁×x₁ + w₂₂×x₂ + b₂
   = 2×1 + 1×2 + 1
   = 2 + 2 + 1
   = 5
```

**Apply ReLU activation:**

```
h₁ = ReLU(z₁) = max(0, 5) = 5
h₂ = ReLU(z₂) = max(0, 5) = 5
```

**Step 3: Output layer computation**

```
ŷ = v₁×h₁ + v₂×h₂ + b_o
  = 2×5 + 1×5 + 0
  = 10 + 5 + 0
  = 15
```

**Step 4: Squared error**

```
E₁ = (1/2)(y - ŷ)²
   = (1/2)(5 - 15)²
   = (1/2)(-10)²
   = (1/2)(100)
   = 50
```

**C₁ Results:**

- **Output: ŷ₁ = 15**
- **Error: E₁ = 50**

---

**CHROMOSOME C₂:**

**Step 1: Extract weights and biases**

```
W₁ = [2  1]
     [1  2]

b₁ = [1]
     [0]

W₂ = [1  2]

b_o = 1
```

**Step 2: Hidden layer computation**

```
z₁ = 2×1 + 1×2 + 1 = 2 + 2 + 1 = 5
z₂ = 1×1 + 2×2 + 0 = 1 + 4 + 0 = 5

h₁ = ReLU(5) = 5
h₂ = ReLU(5) = 5
```

**Step 3: Output layer**

```
ŷ = 1×5 + 2×5 + 1
  = 5 + 10 + 1
  = 16
```

**Step 4: Squared error**

```
E₂ = (1/2)(5 - 16)²
   = (1/2)(-11)²
   = (1/2)(121)
   = 60.5
```

**C₂ Results:**

- **Output: ŷ₂ = 16**
- **Error: E₂ = 60.5**

---

**Summary:**

| Chromosome | Output ŷ | Squared Error E |
| ---------- | -------- | --------------- |
| C₁         | 15       | 50              |
| C₂         | 16       | 60.5            |

---

### Part (b): Select Best Chromosome (1 Mark)

**Answer:**

📝 **Note:** Minimization problem - lower error is better

**Comparison:**

- C₁: Error = 50
- C₂: Error = 60.5

**Answer:** **Chromosome C₁** is selected (minimum error of 50)

---

### Part (c): Backpropagation - One Iteration (4 Marks)

**Answer:**

**Starting with C₁:** [1, 2, 2, 1, 0, 1, 2, 1, 0]

**Current state:**

- h = [5, 5]ᵀ
- ŷ = 15
- Target y = 5
- Error: e = y - ŷ = 5 - 15 = -10

---

**Step 1: Output layer gradients**

📝 **Note:** For linear output neuron:

```
∂E/∂ŷ = -(y - ŷ) = -e
```

**Gradient for output weights:**

```
∂E/∂v₁ = ∂E/∂ŷ × ∂ŷ/∂v₁ = -e × h₁
       = -(-10) × 5
       = 50

∂E/∂v₂ = -e × h₂
       = -(-10) × 5
       = 50
```

**Gradient for output bias:**

```
∂E/∂b_o = -e × 1
        = -(-10)
        = 10
```

---

**Step 2: Hidden layer gradients**

📝 **Note:** Backpropagate error through hidden layer

**Error signal for hidden neurons:**

```
δ₁ = (∂E/∂ŷ) × v₁ × ReLU'(z₁)
   = 10 × 2 × 1  (ReLU' = 1 when z > 0)
   = 20

δ₂ = 10 × 1 × 1
   = 10
```

**Gradients for hidden weights:**

```
∂E/∂w₁₁ = δ₁ × x₁ = 20 × 1 = 20
∂E/∂w₁₂ = δ₁ × x₂ = 20 × 2 = 40
∂E/∂w₂₁ = δ₂ × x₁ = 10 × 1 = 10
∂E/∂w₂₂ = δ₂ × x₂ = 10 × 2 = 20
```

**Gradients for hidden biases:**

```
∂E/∂b₁ = δ₁ = 20
∂E/∂b₂ = δ₂ = 10
```

---

**Step 3: Weight updates (η = 0.1)**

📝 **Note:** Update rule: w_new = w_old - η × ∂E/∂w

**Output layer updates:**

```
v₁_new = v₁ - η × ∂E/∂v₁ = 2 - 0.1 × 50 = 2 - 5 = -3
v₂_new = v₂ - η × ∂E/∂v₂ = 1 - 0.1 × 50 = 1 - 5 = -4
b_o_new = b_o - η × ∂E/∂b_o = 0 - 0.1 × 10 = -1
```

**Hidden layer updates:**

```
w₁₁_new = 1 - 0.1 × 20 = 1 - 2 = -1
w₁₂_new = 2 - 0.1 × 40 = 2 - 4 = -2
w₂₁_new = 2 - 0.1 × 10 = 2 - 1 = 1
w₂₂_new = 1 - 0.1 × 20 = 1 - 2 = -1
b₁_new = 0 - 0.1 × 20 = -2
b₂_new = 1 - 0.1 × 10 = 1 - 1 = 0
```

---

**Updated Chromosome:**

```
C₁_updated = [-1, -2, 1, -1, -2, 0, -3, -4, -1]
```

**Matrix form:**

```
W₁_new = [-1  -2]     b₁_new = [-2]
         [ 1  -1]              [ 0]

W₂_new = [-3  -4]     b_o_new = -1
```

---

### Part (d): Compute Output with Updated Weights (1 Mark)

**Answer:**

**Step 1: Hidden layer with updated weights**

```
z₁ = -1×1 + (-2)×2 + (-2) = -1 - 4 - 2 = -7
z₂ = 1×1 + (-1)×2 + 0 = 1 - 2 + 0 = -1

h₁ = ReLU(-7) = 0
h₂ = ReLU(-1) = 0
```

**Step 2: Output**

```
ŷ_new = -3×0 + (-4)×0 + (-1)
      = 0 + 0 - 1
      = -1
```

**Step 3: New error**

```
E_new = (1/2)(5 - (-1))²
      = (1/2)(6)²
      = (1/2)(36)
      = 18
```

**Results:**

- **New output: ŷ = -1**
- **New error: E = 18**

**Improvement:** Error decreased from 50 to 18 (64% reduction!) ✓

---

**Comment on GA-BP Hybrid:**

📝 **Hybrid Approach Benefits:**

1. **GA Phase:**

   - Global search across weight space
   - Found C₁ as good starting point (E=50)
   - Avoided poor local minima

2. **BP Phase:**

   - Local gradient-based fine-tuning
   - Rapidly reduced error (50 → 18 in one iteration)
   - Efficient convergence

3. **Synergy:**
   - GA provides good initialization
   - BP provides fast refinement
   - **Best of both worlds:** Global exploration + Local exploitation

**Conclusion:** The hybrid GA-BP approach combines GA's ability to find promising regions with BP's gradient-based precision for faster, more reliable convergence than either method alone.

---

## Q10: TRAVELLING SALESMAN PROBLEM WITH GA

### Given Information:

- Number of cities: 10
- Encoding: Links between city pairs (e.g., 'VH' for Vijayawada-Hyderabad)
- Direction not important: VH = HV
- Chromosome: Ordered sequence of cities (tour)

---

### Part (a): Number of Genes per Chromosome (2 Marks)

**Answer:**

📝 **Note:** For TSP, chromosome represents a complete tour

- Tour visits each city exactly once
- Returns to starting city

**Number of cities:** 10

**Chromosome representation:** Ordered list of cities

**Example tour:**

```
Start → City1 → City2 → City3 → ... → City10 → Back to Start
```

**Number of genes = Number of cities = 10**

**Answer:** **10 genes** per chromosome

**Example chromosome:**

```
[3, 7, 2, 9, 1, 5, 8, 4, 10, 6]
```

This represents the tour: Start at city 3, then 7, then 2, ..., then 6, return to 3

---

### Part (b): Genes Required to Encode All City Pairs (3 Marks)

**Answer:**

📝 **Note:** This asks about encoding all possible links, not just one tour

**Total number of unique city pairs:**

For n cities, number of unique undirected pairs:

```
Formula: C(n, 2) = n(n-1)/2
```

**Why?**

- Can choose any 2 cities from 10
- Order doesn't matter (AB = BA)
- This is combinations formula

**Calculation:**

```
Number of pairs = 10 × 9 / 2
                = 90 / 2
                = 45
```

**Answer:** **45 genes** are required to encode all possible city pairs

**Examples of pairs:**

```
City pairs: (1,2), (1,3), (1,4), ..., (1,10)  → 9 pairs with city 1
           (2,3), (2,4), ..., (2,10)          → 8 pairs with city 2
           (3,4), (3,5), ..., (3,10)          → 7 pairs with city 3
           ...
           (9,10)                              → 1 pair
Total: 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 45 ✓
```

---

### Part (c): Fitness Function for TSP (2 Marks)

**Answer:**

**Given:**

- Tour 1: Total route length = 410 km
- Tour 2: Total route length = 380 km

📝 **Note:** TSP is a **minimization** problem (shorter tour is better)

**Fitness function for minimization:**

**Option 1: Inverse fitness**

```
Fitness = 1 / (Total Distance)

Tour 1: f₁ = 1/410 = 0.00244
Tour 2: f₂ = 1/380 = 0.00263

Tour 2 has higher fitness ✓ (better)
```

**Option 2: Negative distance**

```
Fitness = -Total Distance

Tour 1: f₁ = -410
Tour 2: f₂ = -380

Tour 2 has higher fitness ✓ (less negative)
```

**Option 3: Maximum distance minus actual**

```
Fitness = Max_Distance - Total Distance

If max possible = 1000 km:
Tour 1: f₁ = 1000 - 410 = 590
Tour 2: f₂ = 1000 - 380 = 620

Tour 2 has higher fitness ✓
```

---

**Recommended fitness function:**

```
Fitness(tour) = 1 / (1 + Total_Distance)
```

**Advantages:**

- Always positive
- Bounded (max = 1)
- Smooth gradient
- Shorter distance → Higher fitness

**For our tours:**

```
Tour 1: f₁ = 1/(1 + 410) = 1/411 = 0.00243
Tour 2: f₂ = 1/(1 + 380) = 1/381 = 0.00262
```

**Answer:** **Tour 2 (380 km) is fitter** with higher fitness value in all formulations.

---

## Q11: HOPFIELD AND BAM NETWORKS

### Part (a): Hopfield Network

**Given:**

- 4 neurons with bipolar states (+1/-1)
- Training patterns:
  ```
  x₁ = [+1, -1, +1, -1]ᵀ
  x₂ = [-1, +1, -1, +1]ᵀ
  x₃ = [+1, +1, -1, -1]ᵀ
  ```

---

#### Part (a.1): Compute Weight Matrix using Hebbian Learning (2 Marks)

**Answer:**

📝 **Note:** Hebbian learning for Hopfield networks:

```
W = Σ (xᵢ × xᵢᵀ) - M×I

Where:
- M = number of patterns
- I = identity matrix (ensures wᵢᵢ = 0)
- xᵢᵀ = transpose of pattern i
```

**Why?** Weights encode pattern correlations; diagonal = 0 (no self-connections)

---

**Step 1: Compute outer products**

**x₁ × x₁ᵀ:**

```
     [+1]
     [-1]  × [+1  -1  +1  -1]
     [+1]
     [-1]

= [+1  -1  +1  -1]
  [-1  +1  -1  +1]
  [+1  -1  +1  -1]
  [-1  +1  -1  +1]
```

**x₂ × x₂ᵀ:**

```
     [-1]
     [+1]  × [-1  +1  -1  +1]
     [-1]
     [+1]

= [+1  -1  +1  -1]
  [-1  +1  -1  +1]
  [+1  -1  +1  -1]
  [-1  +1  -1  +1]
```

**x₃ × x₃ᵀ:**

```
     [+1]
     [+1]  × [+1  +1  -1  -1]
     [-1]
     [-1]

= [+1  +1  -1  -1]
  [+1  +1  -1  -1]
  [-1  -1  +1  +1]
  [-1  -1  +1  +1]
```

---

**Step 2: Sum outer products**

```
W_total = x₁x₁ᵀ + x₂x₂ᵀ + x₃x₃ᵀ

= [+1  -1  +1  -1]   [+1  -1  +1  -1]   [+1  +1  -1  -1]
  [-1  +1  -1  +1] + [-1  +1  -1  +1] + [+1  +1  -1  -1]
  [+1  -1  +1  -1]   [+1  -1  +1  -1]   [-1  -1  +1  +1]
  [-1  +1  -1  +1]   [-1  +1  -1  +1]   [-1  -1  +1  +1]

= [+3  -1  +1  -3]
  [-1  +3  -3  +1]
  [+1  -3  +3  -1]
  [-3  +1  -1  +3]
```

---

**Step 3: Remove diagonal (set to 0)**

```
W = W_total - 3×I  (M=3 patterns)

= [+3  -1  +1  -3]   [3  0  0  0]
  [-1  +3  -3  +1] - [0  3  0  0]
  [+1  -3  +3  -1]   [0  0  3  0]
  [-3  +1  -1  +3]   [0  0  0  3]

= [0  -1  +1  -3]
  [-1  0  -3  +1]
  [+1  -3  0  -1]
  [-3  +1  -1  0]
```

**Answer:**

```
W = [ 0  -1  +1  -3]
    [-1   0  -3  +1]
    [+1  -3   0  -1]
    [-3  +1  -1   0]
```

---

#### Part (a.2): Synchronous Update for Noisy Pattern (2 Marks)

**Answer:**

**Given noisy input:** x₄ = [+1, -1, -1, -1]ᵀ

📝 **Note:** Synchronous update formula:

```
x_new = sign(W × x_old)

sign(z) = {+1 if z > 0, -1 if z ≤ 0}
```

**Why?** All neurons update simultaneously based on current state

---

**Step 1: Compute W × x₄**

```
[  0  -1  +1  -3] [+1]   [0×1 + (-1)×(-1) + 1×(-1) + (-3)×(-1)]
[ -1   0  -3  +1] [-1] = [(-1)×1 + 0×(-1) + (-3)×(-1) + 1×(-1)]
[ +1  -3   0  -1] [-1]   [1×1 + (-3)×(-1) + 0×(-1) + (-1)×(-1)]
[ -3  +1  -1   0] [-1]   [(-3)×1 + 1×(-1) + (-1)×(-1) + 0×(-1)]

= [0 + 1 - 1 + 3]   [+3]
  [-1 + 0 + 3 - 1]   [+1]
  [1 + 3 + 0 + 1]   [+5]
  [-3 - 1 + 1 + 0]   [-3]
```

---

**Step 2: Apply sign function**

```
x_new = sign([+3, +1, +5, -3]ᵀ)
      = [+1, +1, +1, -1]ᵀ
```

**Answer:** Next state = **[+1, +1, +1, -1]ᵀ**

**Observation:** This matches the stored pattern **x₃ = [+1, +1, -1, -1]**... wait, let me recalculate position 3:

Actually, x₃ = [+1, +1, -1, -1]ᵀ, but we got [+1, +1, +1, -1]ᵀ

Let me verify: The network converged close to x₃ with one bit different.

---

#### Part (a.3): Network Energy Computation (1 Mark)

**Answer:**

📝 **Note:** Energy function for Hopfield network:

```
E(x) = -(1/2) × xᵀ × W × x
```

**Why?** Energy decreases during updates (converges to local minimum)

---

**Energy before update (x₄):**

```
x₄ᵀ × W × x₄ = [+1  -1  -1  -1] × W × x₄

First: W × x₄ = [+3, +1, +5, -3]ᵀ (from previous calculation)

Then: x₄ᵀ × [+3, +1, +5, -3]ᵀ
    = (+1)×(+3) + (-1)×(+1) + (-1)×(+5) + (-1)×(-3)
    = 3 - 1 - 5 + 3
    = 0

E_before = -(1/2) × 0 = 0
```

---

**Energy after update (x_new = [+1, +1, +1, -1]):**

```
W × x_new = [ 0  -1  +1  -3] [+1]
            [-1   0  -3  +1] [+1]
            [+1  -3   0  -1] [+1]
            [-3  +1  -1   0] [-1]

= [0 - 1 + 1 + 3]   [+3]
  [-1 + 0 - 3 - 1]   [-5]
  [1 - 3 + 0 + 1]   [-1]
  [-3 + 1 - 1 + 0]   [-3]

x_newᵀ × W × x_new = [+1  +1  +1  -1] × [+3, -5, -1, -3]ᵀ
                   = 3 - 5 - 1 + 3
                   = 0

E_after = -(1/2) × 0 = 0
```

**Answer:**

- **Energy before:** E = 0
- **Energy after:** E = 0
- **Comment:** Energy remained constant, suggesting the network is at or near equilibrium. The pattern is close to stored pattern x₃.

---

### Part (b): Bidirectional Associative Memory (BAM)

**Given training pairs:**

```
X₁ = [+1, -1, +1, -1]ᵀ,  Y₁ = [+1, -1]ᵀ
X₂ = [-1, +1, -1, +1]ᵀ,  Y₂ = [-1, +1]ᵀ
```

---

#### Part (b.1): Construct Weight Matrix (2 Marks)

**Answer:**

📝 **Note:** BAM weight matrix using Hebbian learning:

```
W = Σ (Xᵢ × Yᵢᵀ)
```

**Why?** W encodes associations between X and Y patterns

---

**Step 1: Compute X₁ × Y₁ᵀ**

```
     [+1]
     [-1]  × [+1  -1]
     [+1]
     [-1]

= [+1  -1]
  [-1  +1]
  [+1  -1]
  [-1  +1]
```

---

**Step 2: Compute X₂ × Y₂ᵀ**

```
     [-1]
     [+1]  × [-1  +1]
     [-1]
     [+1]

= [+1  -1]
  [-1  +1]
  [+1  -1]
  [-1  +1]
```

---

**Step 3: Sum**

```
W = X₁Y₁ᵀ + X₂Y₂ᵀ

= [+1  -1]   [+1  -1]   [+2  -2]
  [-1  +1] + [-1  +1] = [-2  +2]
  [+1  -1]   [+1  -1]   [+2  -2]
  [-1  +1]   [-1  +1]   [-2  +2]
```

**Answer:**

```
W = [+2  -2]
    [-2  +2]
    [+2  -2]
    [-2  +2]
```

---

#### Part (b.2): Forward and Backward Recall (2 Marks)

**Answer:**

**FORWARD RECALL:** X_test = [+1, -1, -1, -1]ᵀ

📝 **Note:** Forward: Y = sign(Wᵀ × X)

**Step 1: Compute Wᵀ × X_test**

```
Wᵀ = [+2  -2  +2  -2]
     [-2  +2  -2  +2]

Wᵀ × X_test = [+2  -2  +2  -2] [+1]
              [-2  +2  -2  +2] [-1]
                              [-1]
                              [-1]

= [2×1 + (-2)×(-1) + 2×(-1) + (-2)×(-1)]
  [(-2)×1 + 2×(-1) + (-2)×(-1) + 2×(-1)]

= [2 + 2 - 2 + 2]   [+4]
  [-2 - 2 + 2 - 2]   [-4]
```

**Step 2: Apply sign**

```
Y_recalled = sign([+4, -4]ᵀ)
           = [+1, -1]ᵀ
```

**Answer:** Forward recall gives **Y = [+1, -1]ᵀ** which matches **Y₁** ✓

---

**BACKWARD RECALL:** Y_test = [+1, +1]ᵀ

📝 **Note:** Backward: X = sign(W × Y)

**Step 1: Compute W × Y_test**

```
W × Y_test = [+2  -2] [+1]
             [-2  +2] [+1]
             [+2  -2]
             [-2  +2]

= [2×1 + (-2)×1]   [ 0]
  [(-2)×1 + 2×1]   [ 0]
  [2×1 + (-2)×1]   [ 0]
  [(-2)×1 + 2×1]   [ 0]
```

**Step 2: Apply sign**

```
X_recalled = sign([0, 0, 0, 0]ᵀ)
           = [-1, -1, -1, -1]ᵀ  (sign(0) = -1 by convention)
```

**Answer:** Backward recall gives **X = [-1, -1, -1, -1]ᵀ**

---

#### Part (b.3): Identify Recalled Patterns (1 Mark)

**Answer:**

**Forward Recall:**

- Input: X_test = [+1, -1, -1, -1]ᵀ
- Output: Y = [+1, -1]ᵀ
- **Matches stored pair:** (X₁, Y₁) → Retrieved Y₁ ✓

**Explanation:** X_test is close to X₁ (3 out of 4 bits match). BAM successfully associated it with the corresponding Y₁.

---

**Backward Recall:**

- Input: Y_test = [+1, +1]ᵀ
- Output: X = [-1, -1, -1, -1]ᵀ
- **Does not match either stored X pattern**

**Explanation:** Y_test = [+1, +1] is equally distant from both Y₁ = [+1, -1] and Y₂ = [-1, +1]. The network produced an ambiguous result (all zeros → all -1s), indicating **no clear association**.

**Conclusion:** BAM successfully recalls stored associations when input is close to a stored pattern, but produces spurious states for ambiguous inputs.

---

## Q12: CNN POOLING OPERATIONS

### Given:

**Feature map A (4×4):**

```
A = [8  4  6  2]
    [3  9  1  7]
    [5  2  8  4]
    [1  6  3  5]
```

**Pooling filter:** 2×2 with stride=2, no padding

📝 **Note:** Stride=2 means non-overlapping windows

- Output size = (4-2)/2 + 1 = 2×2

---

### Part (a): Max Pooling (1 Mark)

**Answer:**

📝 **Note:** Max pooling takes maximum value in each window

- **Why?** Preserves strongest features/activations

**Window divisions:**

```
Window 1 (top-left):    [8  4]  → max = 9
                        [3  9]

Window 2 (top-right):   [6  2]  → max = 7
                        [1  7]

Window 3 (bottom-left): [5  2]  → max = 8
                        [1  6]

Window 4 (bottom-right):[8  4]  → max = 8
                        [3  5]
```

**Max Pooled Output:**

```
[9  7]
[8  8]
```

**Answer:** Max pooling result = **[[9, 7], [8, 8]]**

---

### Part (b): Average Pooling (2 Marks)

**Answer:**

📝 **Note:** Average pooling computes mean of each window

- **Why?** Smooths features, reduces noise

**Window 1 (top-left):**

```
[8  4]
[3  9]

Average = (8 + 4 + 3 + 9) / 4 = 24 / 4 = 6
```

**Window 2 (top-right):**

```
[6  2]
[1  7]

Average = (6 + 2 + 1 + 7) / 4 = 16 / 4 = 4
```

**Window 3 (bottom-left):**

```
[5  2]
[1  6]

Average = (5 + 2 + 1 + 6) / 4 = 14 / 4 = 3.5
```

**Window 4 (bottom-right):**

```
[8  4]
[3  5]

Average = (8 + 4 + 3 + 5) / 4 = 20 / 4 = 5
```

**Average Pooled Output:**

```
[6.0  4.0]
[3.5  5.0]
```

**Answer:** Average pooling result = **[[6.0, 4.0], [3.5, 5.0]]**

---

### Part (c): Min Pooling (1 Mark)

**Answer:**

📝 **Note:** Min pooling takes minimum value in each window

- **Why?** Less common, but useful for detecting absence of features

**Window calculations:**

```
Window 1: [8,4,3,9] → min = 3
Window 2: [6,2,1,7] → min = 1
Window 3: [5,2,1,6] → min = 1
Window 4: [8,4,3,5] → min = 3
```

**Min Pooled Output:**

```
[3  1]
[1  3]
```

**Answer:** Min pooling result = **[[3, 1], [1, 3]]**

---

### Part (d): Translation Invariance (1 Mark)

**Answer:**

📝 **How pooling contributes to translation invariance:**

**Translation Invariance** means the network recognizes objects regardless of their exact position in the image.

**Mechanism:**

1. **Spatial Compression:**
   - Pooling reduces 4×4 to 2×2
   - Small shifts in input position don't change pooled output much
2. **Local Aggregation:**
   - Pooling summarizes local region (2×2 window)
   - If feature shifts slightly within window, output remains same

**Example:**

```
Original:           Shifted right:
[0  8  0  0]       [0  0  8  0]
[0  9  0  0]       [0  0  9  0]

Max pool (2×2):    Max pool (2×2):
Both give 9 in that region!
```

3. **Feature Abstraction:**
   - Max pooling: "Is there a strong activation anywhere in this region?"
   - Average pooling: "What's the overall activation level?"
   - Both abstract away exact spatial location

**Answer:** Pooling achieves translation invariance by:

- **Downsampling** spatial dimensions (reduces sensitivity to exact positions)
- **Local aggregation** (small translations within pooling window produce same output)
- **Feature summarization** (focuses on presence/strength of features rather than precise location)

This allows CNNs to recognize objects even when they appear at different positions in the image.

---

## Q13: CNN LAYER-BY-LAYER ANALYSIS

### Given:

**Input:** RGB images 128×128×3

**Layers:**

1. Conv 3×3, stride=1, padding=1, C_out=32
2. Conv 3×3, stride=2, padding=1, C_out=64
3. MaxPool 2×2, stride=2
4. Global Average Pool
5. Fully Connected 64→10

---

### Solution:

📝 **Formulas needed:**

**Output dimension:**

```
H_out = (H_in + 2×p - k) / s + 1
W_out = (W_in + 2×p - k) / s + 1
```

**Parameters (Conv):**

```
Params = (k × k × C_in + 1) × C_out
```

(+1 for bias per output channel)

**FLOPs (Conv):**

```
FLOPs = H_out × W_out × C_out × (k × k × C_in + 1)
```

**FLOPs (FC):**

```
FLOPs = Input_size × Output_size
```

---

### Layer 1: Conv 3×3, s=1, p=1, C_out=32

**Input:** 128×128×3

**Output dimensions:**

```
H_out = (128 + 2×1 - 3)/1 + 1 = 128
W_out = (128 + 2×1 - 3)/1 + 1 = 128
C_out = 32
```

**Output:** **128×128×32**

**Parameters:**

```
Params = (3 × 3 × 3 + 1) × 32
       = (27 + 1) × 32
       = 28 × 32
       = 896
```

**FLOPs:**

```
FLOPs = 128 × 128 × 32 × (3 × 3 × 3 + 1)
      = 128 × 128 × 32 × 28
      = 14,680,064
      ≈ 14.68M
```

---

### Layer 2: Conv 3×3, s=2, p=1, C_out=64

**Input:** 128×128×32

**Output dimensions:**

```
H_out = (128 + 2×1 - 3)/2 + 1 = 64
W_out = (128 + 2×1 - 3)/2 + 1 = 64
C_out = 64
```

**Output:** **64×64×64**

**Parameters:**

```
Params = (3 × 3 × 32 + 1) × 64
       = (288 + 1) × 64
       = 289 × 64
       = 18,496
```

**FLOPs:**

```
FLOPs = 64 × 64 × 64 × (3 × 3 × 32 + 1)
      = 64 × 64 × 64 × 289
      = 75,497,472
      ≈ 75.50M
```

---

### Layer 3: MaxPool 2×2, s=2

**Input:** 64×64×64

📝 **Note:** Pooling has NO parameters (no learnable weights)

**Output dimensions:**

```
H_out = (64 - 2)/2 + 1 = 32
W_out = (64 - 2)/2 + 1 = 32
C_out = 64 (unchanged)
```

**Output:** **32×32×64**

**Parameters:** **0**

**FLOPs:**

```
FLOPs = H_out × W_out × C_out × (pool_size²)
      = 32 × 32 × 64 × 4
      = 262,144
      ≈ 0.26M
```

(4 comparisons per 2×2 window for max operation)

---

### Layer 4: Global Average Pool

**Input:** 32×32×64

📝 **Note:** Global Average Pool averages each channel spatially

- Output: 1×1×C (one value per channel)

**Output dimensions:** **1×1×64** or just **64×1**

**Parameters:** **0**

**FLOPs:**

```
FLOPs = C × (H × W)  (sum then divide for each channel)
      = 64 × (32 × 32)
      = 64 × 1024
      = 65,536
      ≈ 0.07M
```

---

### Layer 5: Fully Connected 64→10

**Input:** 64×1

**Output:** **10×1**

**Parameters:**

```
Params = (Input_size + 1) × Output_size
       = (64 + 1) × 10
       = 65 × 10
       = 650
```

(+1 for bias)

**FLOPs:**

```
FLOPs = Input_size × Output_size
      = 64 × 10
      = 640
```

---

### Complete Table:

| Layer                           | Output Dimension | # Parameters | # Operations (FLOPs) |
| ------------------------------- | ---------------- | ------------ | -------------------- |
| **Input**                       | 128×128×3        | -            | -                    |
| **1: Conv 3×3, s=1, p=1, C=32** | 128×128×32       | 896          | 14.68M               |
| **2: Conv 3×3, s=2, p=1, C=64** | 64×64×64         | 18,496       | 75.50M               |
| **3: MaxPool 2×2, s=2**         | 32×32×64         | 0            | 0.26M                |
| **4: Global Avg Pool**          | 64×1             | 0            | 0.07M                |
| **5: FC 64→10**                 | 10×1             | 650          | 0.00064M             |
| **TOTALS**                      | -                | **20,042**   | **90.51M**           |

---

**Analysis:**

1. **Most parameters:** Layer 2 (Conv 64 filters) with 18,496 params
2. **Most FLOPs:** Layer 2 with 75.50M operations (83% of total)
3. **Pooling layers:** No parameters, minimal computation
4. **FC layer:** Minimal cost due to Global Avg Pool reducing dimensions

---

## Q14: BACKPROPAGATION - DETAILED CALCULATION

### Given:

**Network:** 3 input → 2 hidden (ReLU) → 1 output (linear)

**Data:**

```
x = [2, 1, 3]ᵀ

W₁ = [1  0  2]    b₁ = [1]
     [0  1  1]         [-1]

W₂ = [2  1]       b₂ = 1

Target: d = 8
Learning rate: η = 0.5
Loss: L = (1/2)(d - y)²
```

---

### Part 1: Compute Output y (2 Marks)

**Answer:**

**Step 1: Hidden layer computation**

```
z₁ = w₁₁×x₁ + w₁₂×x₂ + w₁₃×x₃ + b₁
   = 1×2 + 0×1 + 2×3 + 1
   = 2 + 0 + 6 + 1
   = 9

z₂ = w₂₁×x₁ + w₂₂×x₂ + w₂₃×x₃ + b₂
   = 0×2 + 1×1 + 1×3 + (-1)
   = 0 + 1 + 3 - 1
   = 3
```

**Apply ReLU:**

```
h₁ = ReLU(z₁) = max(0, 9) = 9
h₂ = ReLU(z₂) = max(0, 3) = 3
```

**Step 2: Output layer**

```
y = w₂₁×h₁ + w₂₂×h₂ + b₂
  = 2×9 + 1×3 + 1
  = 18 + 3 + 1
  = 22
```

**Answer: y = 22**

---

### Part 2: Compute Loss L (1 Mark)

**Answer:**

```
L = (1/2)(d - y)²
  = (1/2)(8 - 22)²
  = (1/2)(-14)²
  = (1/2)(196)
  = 98
```

**Answer: L = 98**

---

### Part 3: Compute Gradients (5 Marks)

**Answer:**

📝 **Note:** Use chain rule systematically from output to input

---

**LAYER 2 GRADIENTS (Output Layer):**

**Error signal:**

```
∂L/∂y = -(d - y) = -(8 - 22) = 14
```

**Gradient for W₂:**

```
∂L/∂W₂ = ∂L/∂y × ∂y/∂W₂
       = 14 × h

∂L/∂w₂₁ = 14 × h₁ = 14 × 9 = 126
∂L/∂w₂₂ = 14 × h₂ = 14 × 3 = 42
```

**Gradient for b₂:**

```
∂L/∂b₂ = ∂L/∂y × ∂y/∂b₂
       = 14 × 1
       = 14
```

---

**LAYER 1 GRADIENTS (Hidden Layer):**

**Backpropagate error:**

```
∂L/∂h₁ = ∂L/∂y × ∂y/∂h₁ = 14 × w₂₁ = 14 × 2 = 28
∂L/∂h₂ = ∂L/∂y × ∂y/∂h₂ = 14 × w₂₂ = 14 × 1 = 14
```

**Apply ReLU derivative:**

```
ReLU'(z) = {1 if z > 0, 0 if z ≤ 0}

Since z₁ = 9 > 0: ReLU'(z₁) = 1
Since z₂ = 3 > 0: ReLU'(z₂) = 1

∂L/∂z₁ = ∂L/∂h₁ × ReLU'(z₁) = 28 × 1 = 28
∂L/∂z₂ = ∂L/∂h₂ × ReLU'(z₂) = 14 × 1 = 14
```

**Gradient for W₁:**

```
∂L/∂W₁ = ∂L/∂z × xᵀ

For first row (neuron 1):
∂L/∂w₁₁ = ∂L/∂z₁ × x₁ = 28 × 2 = 56
∂L/∂w₁₂ = ∂L/∂z₁ × x₂ = 28 × 1 = 28
∂L/∂w₁₃ = ∂L/∂z₁ × x₃ = 28 × 3 = 84

For second row (neuron 2):
∂L/∂w₂₁ = ∂L/∂z₂ × x₁ = 14 × 2 = 28
∂L/∂w₂₂ = ∂L/∂z₂ × x₂ = 14 × 1 = 14
∂L/∂w₂₃ = ∂L/∂z₂ × x₃ = 14 × 3 = 42
```

**Matrix form:**

```
∂L/∂W₁ = [56  28  84]
         [28  14  42]
```

**Gradient for b₁:**

```
∂L/∂b₁ = ∂L/∂z₁ = 28
∂L/∂b₂ = ∂L/∂z₂ = 14

∂L/∂b₁ = [28]
         [14]
```

---

**Summary of All Gradients:**

```
∂L/∂W₂ = [126  42]
∂L/∂b₂ = 14

∂L/∂W₁ = [56  28  84]
         [28  14  42]

∂L/∂b₁ = [28]
         [14]
```

---

### Part 4: Weight Update (1 Mark)

**Answer:**

📝 **Update rule:** w_new = w_old - η × ∂L/∂w

**Layer 2 updates (η = 0.5):**

```
W₂_new = W₂ - 0.5 × ∂L/∂W₂
       = [2  1] - 0.5 × [126  42]
       = [2  1] - [63  21]
       = [-61  -20]

b₂_new = 1 - 0.5 × 14
       = 1 - 7
       = -6
```

**Layer 1 updates:**

```
W₁_new = W₁ - 0.5 × ∂L/∂W₁
       = [1  0  2] - 0.5 × [56  28  84]
         [0  1  1]         [28  14  42]

       = [1  0  2] - [28  14  42]
         [0  1  1]   [14   7  21]

       = [-27  -14  -40]
         [-14   -6  -20]

b₁_new = [1]  - 0.5 × [28]
         [-1]         [14]

       = [1]  - [14]
         [-1]   [7]

       = [-13]
         [-8]
```

**Answer:**

**Updated parameters:**

```
W₁ = [-27  -14  -40]    b₁ = [-13]
     [-14   -6  -20]         [-8]

W₂ = [-61  -20]         b₂ = -6
```

---

### Part 5: Compute Output with Updated Weights (1 Mark)

**Answer:**

**Step 1: Hidden layer with new weights**

```
z₁ = -27×2 + (-14)×1 + (-40)×3 + (-13)
   = -54 - 14 - 120 - 13
   = -201

z₂ = -14×2 + (-6)×1 + (-20)×3 + (-8)
   = -28 - 6 - 60 - 8
   = -102

h₁ = ReLU(-201) = 0
h₂ = ReLU(-102) = 0
```

**Step 2: Output**

```
y_new = -61×0 + (-20)×0 + (-6)
      = 0 + 0 - 6
      = -6
```

**Step 3: New loss**

```
L_new = (1/2)(8 - (-6))²
      = (1/2)(14)²
      = (1/2)(196)
      = 98
```

**Results:**

- **New output: y = -6**
- **New loss: L = 98**

**Observation:** Loss remained the same (98), but output moved from 22 to -6, getting closer to target 8. With such a large learning rate (0.5), the network overshot and needs more iterations to converge.

---

**End of Solutions**

---

**Summary:**

This document covered all 14 questions with complete step-by-step solutions including:

- Fuzzy logic systems (membership functions, operations)
- Genetic algorithms (fitness, selection, crossover)
- GA-Fuzzy hybrid controllers
- Neural networks (Hopfield, BAM, feedforward)
- CNN operations (pooling, layer analysis)
- Detailed backpropagation calculations

**Exam Tips:**

1. Always show formulas before calculations
2. Label all intermediate steps
3. Verify dimensions in matrix operations
4. Check final answers for reasonableness
5. Include brief explanations for "why" formulas work

**Good luck with your exam!** 🎓
