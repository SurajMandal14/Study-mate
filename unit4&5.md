# Genetic Algorithms and Fuzzy Logic - Numerical Solutions (Units 4 & 5)

This document provides detailed, student-friendly solutions to numerical examples and exercises from Units 4 and 5 of the "Unit-4 and 5.pdf" document. Each solution includes the relevant formulas and a brief explanation to enhance understanding.

---

## Unit 4: Core Concepts in Genetic Algorithms

### 1. Gene

#### Numerical Examples (1.5)

**Example 1: Binary gene for "Use feature?"**
*g = (0: Do not use feature, 1: Use feature)*

**Example 2: Integer gene for "Number of workers"**
*g ∈ {1, 2, 3, 4, 5} with current value g = 3*

#### Exercise Questions (1.6)

**Q1. Define a gene for "Room temperature setting" with possible values: 18°C, 20°C, 22°C, 24°C, 26°C.**
*Solution:*
A gene `g_temp` can be defined with the domain `S_temp = {18°C, 20°C, 22°C, 24°C, 26°C}`. This gene represents a single adjustable parameter for room temperature.

**Q2. Create a binary gene for "Include backup system?" with Yes/No options.**
*Solution:*
A binary gene `g_backup` can be defined as `g_backup = (0: No, 1: Yes)`. Here, 0 represents not including a backup system and 1 represents including it.

**Q3. Design a gene for "Material type" with options: Steel, Aluminum, Carbon Fiber, Titanium.**
*Solution:*
A categorical gene `g_material` can be defined with the domain `S_material = {Steel, Aluminum, Carbon Fiber, Titanium}`. Each value represents a different material choice.

**Q4. For a gene representing "Engine size" from 1000cc to 2000cc in 100cc increments, how many possible values exist?**
*Solution:*
*Formula:* Number of values = (Maximum Value - Minimum Value) / Increment + 1
*Calculation:*
`Number of values = (2000 - 1000) / 100 + 1 = 1000 / 100 + 1 = 10 + 1 = 11`
There are 11 possible values for the engine size gene.

**Q5. Calculate the number of bits needed to represent a gene with 16 possible values.**
*Solution:*
*Formula:* Number of bits (n) such that 2^n >= Number of possible values
*Calculation:*
We need to find 'n' such that `2^n >= 16`.
`2^1 = 2`
`2^2 = 4`
`2^3 = 8`
`2^4 = 16`
Therefore, 4 bits are needed to represent a gene with 16 possible values.

#### 10-Mark Question (Gene)

**Question:** Explain the concept of a 'gene' in the context of Genetic Algorithms, providing its motivation and intuitive understanding. Illustrate with at least three diverse examples. Subsequently, consider a scenario where you need to represent the 'speed setting' of a robotic arm, with options ranging from 'Slow', 'Medium', 'Fast', to 'Very Fast'. Design a suitable gene for this, determine the number of bits required for its binary representation, and explain your choice.

---

### 2. Chromosome

#### Numerical Examples (2.5)

**Example 1: Simple 3-gene chromosome (binary)**
*C = [1, 0, 1] representing: Feature1=ON, Feature2=OFF, Feature3=ON*

**Example 2: Real-valued chromosome**
*C = [0.5, 2.3, 45] where: width=0.5m, height=2.3m, angle=45°*

#### Exercise Questions (2.6)

**Q1. Create a chromosome for a "Smart Home" with genes: Temperature (22°C), Lights (ON), Security (OFF), Music (Jazz).**
*Solution:*
A chromosome `C_smart_home` can be represented as an ordered tuple of these genes:
`C_smart_home = [Temperature: 22°C, Lights: ON, Security: OFF, Music: Jazz]`
This chromosome represents a complete configuration for the smart home system.

**Q2. Design a chromosome for a "Study Schedule" with: Hours (4), Break (30min), Time (Morning), Subject Order (Rotating).**
*Solution:*
A chromosome `C_study_schedule` can be represented as:
`C_study_schedule = [Hours: 4, Break: 30min, Time: Morning, Subject Order: Rotating]`
This represents a complete study schedule plan.

**Q3. If a chromosome has 5 genes, each with 4 possible values, how many different chromosomes are possible?**
*Solution:*
*Formula:* Total possible chromosomes = (Number of values per gene) ^ (Number of genes)
*Calculation:*
`Total possible chromosomes = 4 ^ 5 = 4 * 4 * 4 * 4 * 4 = 1024`
There are 1024 different chromosomes possible.

**Q4. Convert the human chromosome [Female, Medium, Brown eyes] to binary if: Female=0, Male=1; Short=00, Medium=01, Tall=10, Very Tall=11; Brown=00, Blue=01, Green=10, Hazel=11.**
*Solution:*
*Mapping each gene to binary:*
`Female` -> `0`
`Medium` -> `01`
`Brown eyes` -> `00`
*Concatenating the binary representations:*
`Binary Chromosome = 00100`

**Q5. A bridge design chromosome has genes: width (3 bits), material (2 bits), angle (3 bits). What is the total chromosome length in bits?**
*Solution:*
*Formula:* Total chromosome length = Sum of bits for each gene
*Calculation:*
`Total chromosome length = 3 bits (width) + 2 bits (material) + 3 bits (angle) = 8 bits`
The total chromosome length is 8 bits.

#### 10-Mark Question (Chromosome)

**Question:** Define a 'chromosome' in the context of Genetic Algorithms, highlighting its role as a complete solution blueprint. Provide an intuitive analogy. Given a system for optimizing traffic light timings, where each intersection's light cycle is controlled by three genes: `Green Light Duration (20-60 seconds in 10-second steps)`, `Yellow Light Duration (3-5 seconds in 1-second steps)`, and `Red Light Duration (20-60 seconds in 10-second steps)`. Design a chromosome for a single intersection. If there are 10 such intersections, explain how you would represent the entire city's traffic light system as a single chromosome, and calculate the total number of possible unique solutions for one intersection.

---

### 3. Population

#### Numerical Examples (3.5)

**Example 1: Small population (size 3)**
*P = {[1, 0, 1], [0, 1, 0], [1, 1, 0]}*

**Example 2: Population statistics**
*Population size N = 100*
*Chromosome length L = 20 bits*
*Total bits considered = N × L = 2000*

#### Exercise Questions (3.6)

**Q1. Create a population of size 4 for a 3-gene system where each gene is binary (0 or 1).**
*Solution:*
*Example Population:*
`P = {`
`  [0, 0, 0],`
`  [1, 0, 1],`
`  [0, 1, 1],`
`  [1, 1, 0]`
`}`
(Any 4 unique or non-unique 3-gene binary chromosomes would suffice as an example.)

**Q2. If each chromosome has 8 bits and population size is 50, how many total bits are being processed?**
*Solution:*
*Formula:* Total bits processed = Population Size × Chromosome Length (in bits)
*Calculation:*
`Total bits processed = 50 × 8 = 400 bits`

**Q3. What happens to exploration capability if population size is too small (e.g., N=2)?**
*Solution:*
If the population size is too small, the **exploration capability decreases significantly**. A small population means less diversity in the genetic material (chromosomes). This can lead to: 
*   **Premature convergence:** The algorithm might quickly converge to a sub-optimal solution (local optimum) because there isn't enough variety to explore other parts of the search space. 
*   **Reduced genetic diversity:** Genetic operations like crossover and mutation have less unique material to work with, limiting the generation of novel solutions.

**Q4. Calculate population diversity if all chromosomes are identical vs all are completely different.**
*Solution:*
*   **All chromosomes are identical:** Diversity is **zero** (or very low). There is no genetic variation within the population. 
*   **All chromosomes are completely different:** Diversity is **maximal** (or very high). Each individual represents a unique point in the search space, maximizing exploration potential. (Note: A more precise calculation would involve Hamming distance or other diversity measures, but qualitatively, this is the interpretation).

**Q5. For a problem with search space size 2^20, what percentage is explored by a population of size 100?**
*Solution:*
*Formula:* Percentage explored = (Population Size / Search Space Size) × 100%
*Calculation:*
`Search Space Size = 2^20 = 1,048,576`
`Percentage explored = (100 / 1,048,576) × 100% ≈ 0.0095%`
A population of size 100 explores a very small percentage of the total search space in this scenario. This highlights that GAs don't exhaustively search but rather intelligently explore a subset.

#### 10-Mark Question (Population)

**Question:** Discuss the significance of 'population' in Genetic Algorithms, drawing parallels with natural populations. Explain how population size influences the trade-off between exploration and exploitation. Consider a scenario where you are optimizing the placement of 5 components on a circuit board, and each component can be placed in 10 distinct locations. If a chromosome represents one complete placement configuration, suggest an appropriate population size, justify your choice, and explain the potential implications of choosing a population size that is too large or too small for this problem.

---

### 4. Encoding

#### Numerical Examples (4.5)

**Example 1: Encode 25°C where range = [20,30], n=4 bits:**
*Formula for k:* `k = round[((x - a) / (b - a)) * (2^n - 1)]`
*Calculation:*
`k = round[((25 - 20) / (30 - 20)) * (2^4 - 1)]`
`k = round[(5 / 10) * (16 - 1)]`
`k = round[0.5 * 15] = round[7.5] = 8`
*Binary encoding:*
`Binary(8, 4 bits) = 1000`

**Example 2: Decode 0110 where range = [0,100], n=4 bits:**
*Formula for k:* `k = binary_to_decimal(binary_string)`
*Formula for x:* `x = a + k * ((b - a) / (2^n - 1))`
*Calculation:*
`k = binary_to_decimal(0110) = 6`
`x = 0 + 6 * ((100 - 0) / (2^4 - 1))`
`x = 0 + 6 * (100 / 15) = 6 * 6.67 ≈ 40`

#### Exercise Questions (4.6)

**Q1. Encode pressure 750 mmHg where range = [700,800] mmHg using 5 bits.**
*Solution:*
*Given:* `x = 750`, `range = [700, 800]`, `n = 5 bits`
*Formula for k:* `k = round[((x - a) / (b - a)) * (2^n - 1)]`
*Calculation:*
`k = round[((750 - 700) / (800 - 700)) * (2^5 - 1)]`
`k = round[(50 / 100) * (32 - 1)]`
`k = round[0.5 * 31] = round[15.5] = 16`
*Binary encoding:*
`Binary(16, 5 bits) = 10000`

**Q2. Decode binary 10101 for range [0,1] with 5-bit precision.**
*Solution:*
*Given:* `binary_string = 10101`, `range = [0, 1]`, `n = 5 bits`
*Formula for k:* `k = binary_to_decimal(binary_string)`
*Calculation of k:*
`k = 1*2^4 + 0*2^3 + 1*2^2 + 0*2^1 + 1*2^0 = 16 + 0 + 4 + 0 + 1 = 21`
*Formula for x:* `x = a + k * ((b - a) / (2^n - 1))`
*Calculation of x:*
`x = 0 + 21 * ((1 - 0) / (2^5 - 1))`
`x = 21 * (1 / 31) ≈ 0.6774`

**Q3. Design encoding for "Speed level" with: Slow, Medium, Fast, Very Fast.**
*Solution:*
This is a categorical encoding. Since there are 4 possible values, we need `log2(4) = 2` bits.
*Example Encoding:*
`Slow = 00`
`Medium = 01`
`Fast = 10`
`Very Fast = 11`

**Q4. What’s the precision (smallest difference) for 8-bit encoding of range [0,255]?**
*Solution:*
*Formula:* Precision = (Range Max - Range Min) / (2^n - 1)
*Calculation:*
`Precision = (255 - 0) / (2^8 - 1)`
`Precision = 255 / (256 - 1) = 255 / 255 = 1`
The smallest difference (precision) is 1. This means each integer value from 0 to 255 can be represented exactly.

**Q5. Encode a person: Female (0), Tall (11), Blue eyes (01), Blood A (00) into single binary string.**
*Solution:*
*Mapping each trait to binary:*
`Female` -> `0` (1 bit)
`Tall` -> `11` (2 bits)
`Blue eyes` -> `01` (2 bits)
`Blood A` -> `00` (2 bits)
*Concatenating the binary representations:*
`Binary String = 0110100`

#### 10-Mark Question (Encoding)

**Question:** Explain the concept and importance of 'encoding' in Genetic Algorithms, providing intuitive examples. Describe how a real-valued parameter within a specified range can be converted into a binary string for genetic operations and then decoded back. Consider a sensor reading for 'humidity' that can range from 0% to 100%. You need to encode this value using 7 bits for a GA. Show the step-by-step encoding of a humidity reading of 65% and the decoding of the binary string `0110100` back into a humidity value, clearly stating the formulas used.

---

### 5. Fitness Function

#### Numerical Examples (5.5)

**Example 1: Maximize test score:**
*f(C) = Number of correct answers - 0.5 × Time taken (hours)*
*If C has 8 correct in 2 hours:*
*Calculation:*
`f = 8 - 0.5 × 2 = 8 - 1 = 7`

**Example 2: Minimize cost with constraint:**
*f(C) = 1 / (Cost + 1000 × max(0, Quality - 80))*
*If Cost=$500, Quality=85:*
*Calculation:*
`f = 1 / (500 + 1000 × max(0, 85 - 80))`
`f = 1 / (500 + 1000 × 5)`
`f = 1 / (500 + 5000)`
`f = 1 / 5500 ≈ 0.00018`

#### Exercise Questions (5.6)

**Q1. Design fitness for basketball player: f = 0.6 × Height + 0.4 × Accuracy.**
*Solution:*
This fitness function aims to maximize a player's combined score based on height and accuracy, with height being slightly more important.
*Formula: `f = 0.6 * Height + 0.4 * Accuracy`*
(Note: Height and Accuracy would typically be normalized or within a defined range for meaningful comparison.)

**Q2. Calculate fitness for C = [Temp = 22, Humidity = 60] if f = 100 - |Temp - 21| - 0.5 × |Humidity - 50|.**
*Solution:*
*Given:* `Temp = 22`, `Humidity = 60`
*Formula: `f = 100 - |Temp - 21| - 0.5 × |Humidity - 50|`*
*Calculation:*
`f = 100 - |22 - 21| - 0.5 × |60 - 50|`
`f = 100 - |1| - 0.5 × |10|`
`f = 100 - 1 - 0.5 × 10`
`f = 100 - 1 - 5 = 94`

**Q3. Create fitness for diet: Maximize protein, minimize calories: f = Protein - 0.1 × Calories.**
*Solution:*
This fitness function attempts to find a diet with high protein and low calories. The `0.1` factor scales down the impact of calories relative to protein.
*Formula: `f = Protein - 0.1 * Calories`*
(Note: Protein and Calories values would need to be in consistent units or scaled appropriately.)

**Q4. Add penalty for budget overrun: If budget=$1000, actual=$1200, penalty coefficient=10.**
*Solution:*
*Concept:* A penalty term is added to the fitness function when a constraint (like budget) is violated. The `max(0, actual - budget)` ensures a penalty is only applied when the actual cost exceeds the budget.
*Formula:* Assuming a base fitness `f_base`, the penalized fitness could be:
`f_penalized = f_base - penalty_coefficient × max(0, actual_cost - budget)`
*Calculation of Penalty Term:*
`Penalty Term = 10 × max(0, 1200 - 1000)`
`Penalty Term = 10 × max(0, 200)`
`Penalty Term = 10 × 200 = 2000`
(This `2000` would be subtracted from the `f_base` fitness.)

**Q5. Normalize fitness scores [5, 15, 10] to range [0, 100].**
*Solution:*
*Given Scores:* `[5, 15, 10]`
*Min Score = 5*
*Max Score = 15*
*Formula for Min-Max Normalization (to range [new_min, new_max]):*
`Normalized_Score = ((Score - Min_Score) / (Max_Score - Min_Score)) * (new_max - new_min) + new_min`
Here, `new_min = 0`, `new_max = 100`.

*Calculation:*
*For Score = 5:*
`Normalized_5 = ((5 - 5) / (15 - 5)) * (100 - 0) + 0`
`Normalized_5 = (0 / 10) * 100 = 0`

*For Score = 15:*
`Normalized_15 = ((15 - 5) / (15 - 5)) * (100 - 0) + 0`
`Normalized_15 = (10 / 10) * 100 = 100`

*For Score = 10:*
`Normalized_10 = ((10 - 5) / (15 - 5)) * (100 - 0) + 0`
`Normalized_10 = (5 / 10) * 100 = 0.5 * 100 = 50`

*Normalized Scores: [0, 100, 50]*

#### 10-Mark Question (Fitness Function)

**Question:** Elaborate on the role and importance of the 'fitness function' in Genetic Algorithms, explaining its purpose as a
