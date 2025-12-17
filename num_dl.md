# DEEP LEARNING - NUMERICAL PROBLEMS SOLUTIONS

**Exam-Ready Solutions with Step-by-Step Calculations**

---

## **Question 1: Convolution + ReLU + MaxPooling (5 Marks)**

### **Given:**

**Input Image (5×5):**

```
[1  2  1  2  1]
[2  1  1  1  2]
[2  2  3  2  3]
[3  2  3  1  2]
[1  2  1  2  1]
```

**Convolution Kernel (3×3):**

```
[ 1   0   1]
[ 1   0   1]
[-1   0  -1]
```

**Task:**

1. Apply convolution
2. Apply ReLU activation
3. Apply 2×2 max pooling

---

### **Step 1: Convolution Operation**

**Formula for Output Size:**
$$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$$

Where:

- $H_{in}$ = 5 (input height)
- $K$ = 3 (kernel size)
- $P$ = 0 (no padding specified)
- $S$ = 1 (stride = 1, default)

**Output size:**
$$H_{out} = \frac{5 - 3 + 0}{1} + 1 = 3$$

**Result:** 3×3 output after convolution

---

### **Convolution Calculations:**

**How convolution works:**

- Slide kernel over input
- Element-wise multiplication
- Sum all products

**Position (0,0) - Top-Left:**

```
Input patch:          Kernel:
[1  2  1]             [ 1   0   1]
[2  1  1]       ⊙     [ 1   0   1]
[2  2  3]             [-1   0  -1]

Computation:
(1×1) + (2×0) + (1×1) +
(2×1) + (1×0) + (1×1) +
(2×-1) + (2×0) + (3×-1)
= 1 + 0 + 1 + 2 + 0 + 1 + (-2) + 0 + (-3)
= 1 + 1 + 2 + 1 - 2 - 3
= 0

Output[0,0] = 0
```

**Position (0,1) - Top-Middle:**

```
Input patch:          Kernel:
[2  1  2]             [ 1   0   1]
[1  1  1]       ⊙     [ 1   0   1]
[2  3  2]             [-1   0  -1]

Computation:
(2×1) + (1×0) + (2×1) +
(1×1) + (1×0) + (1×1) +
(2×-1) + (3×0) + (2×-1)
= 2 + 0 + 2 + 1 + 0 + 1 + (-2) + 0 + (-2)
= 2 + 2 + 1 + 1 - 2 - 2
= 2

Output[0,1] = 2
```

**Position (0,2) - Top-Right:**

```
Input patch:          Kernel:
[1  2  1]             [ 1   0   1]
[1  1  2]       ⊙     [ 1   0   1]
[3  2  3]             [-1   0  -1]

Computation:
(1×1) + (2×0) + (1×1) +
(1×1) + (1×0) + (2×1) +
(3×-1) + (2×0) + (3×-1)
= 1 + 0 + 1 + 1 + 0 + 2 + (-3) + 0 + (-3)
= 1 + 1 + 1 + 2 - 3 - 3
= -1

Output[0,2] = -1
```

**Position (1,0) - Middle-Left:**

```
Input patch:          Kernel:
[2  1  1]             [ 1   0   1]
[2  2  3]       ⊙     [ 1   0   1]
[3  2  3]             [-1   0  -1]

Computation:
(2×1) + (1×0) + (1×1) +
(2×1) + (2×0) + (3×1) +
(3×-1) + (2×0) + (3×-1)
= 2 + 0 + 1 + 2 + 0 + 3 + (-3) + 0 + (-3)
= 2 + 1 + 2 + 3 - 3 - 3
= 2

Output[1,0] = 2
```

**Position (1,1) - Center:**

```
Input patch:          Kernel:
[1  1  1]             [ 1   0   1]
[2  3  2]       ⊙     [ 1   0   1]
[2  3  1]             [-1   0  -1]

Computation:
(1×1) + (1×0) + (1×1) +
(2×1) + (3×0) + (2×1) +
(2×-1) + (3×0) + (1×-1)
= 1 + 0 + 1 + 2 + 0 + 2 + (-2) + 0 + (-1)
= 1 + 1 + 2 + 2 - 2 - 1
= 3

Output[1,1] = 3
```

**Position (1,2) - Middle-Right:**

```
Input patch:          Kernel:
[1  1  2]             [ 1   0   1]
[3  2  3]       ⊙     [ 1   0   1]
[3  1  2]             [-1   0  -1]

Computation:
(1×1) + (1×0) + (2×1) +
(3×1) + (2×0) + (3×1) +
(3×-1) + (1×0) + (2×-1)
= 1 + 0 + 2 + 3 + 0 + 3 + (-3) + 0 + (-2)
= 1 + 2 + 3 + 3 - 3 - 2
= 4

Output[1,2] = 4
```

**Position (2,0) - Bottom-Left:**

```
Input patch:          Kernel:
[2  2  3]             [ 1   0   1]
[3  2  3]       ⊙     [ 1   0   1]
[1  2  1]             [-1   0  -1]

Computation:
(2×1) + (2×0) + (3×1) +
(3×1) + (2×0) + (3×1) +
(1×-1) + (2×0) + (1×-1)
= 2 + 0 + 3 + 3 + 0 + 3 + (-1) + 0 + (-1)
= 2 + 3 + 3 + 3 - 1 - 1
= 9

Output[2,0] = 9
```

**Position (2,1) - Bottom-Middle:**

```
Input patch:          Kernel:
[2  3  2]             [ 1   0   1]
[2  3  1]       ⊙     [ 1   0   1]
[2  1  2]             [-1   0  -1]

Computation:
(2×1) + (3×0) + (2×1) +
(2×1) + (3×0) + (1×1) +
(2×-1) + (1×0) + (2×-1)
= 2 + 0 + 2 + 2 + 0 + 1 + (-2) + 0 + (-2)
= 2 + 2 + 2 + 1 - 2 - 2
= 3

Output[2,1] = 3
```

**Position (2,2) - Bottom-Right:**

```
Input patch:          Kernel:
[3  2  3]             [ 1   0   1]
[3  1  2]       ⊙     [ 1   0   1]
[1  2  1]             [-1   0  -1]

Computation:
(3×1) + (2×0) + (3×1) +
(3×1) + (1×0) + (2×1) +
(1×-1) + (2×0) + (1×-1)
= 3 + 0 + 3 + 3 + 0 + 2 + (-1) + 0 + (-1)
= 3 + 3 + 3 + 2 - 1 - 1
= 9

Output[2,2] = 9
```

**Convolution Output (3×3):**

```
[ 0   2  -1]
[ 2   3   4]
[ 9   3   9]
```

---

### **Step 2: Apply ReLU Activation**

**ReLU Formula:**
$$\text{ReLU}(x) = \max(0, x)$$

- If x > 0 → Keep value
- If x ≤ 0 → Set to 0

**Applying ReLU:**

```
Before ReLU:          After ReLU:
[ 0   2  -1]         [ 0   2   0]
[ 2   3   4]    →    [ 2   3   4]
[ 9   3   9]         [ 9   3   9]
```

**Explanation:**

- Position [0,0]: 0 → max(0, 0) = 0
- Position [0,1]: 2 → max(0, 2) = 2
- Position [0,2]: -1 → max(0, -1) = 0 (negative removed!)
- Position [1,0]: 2 → max(0, 2) = 2
- Position [1,1]: 3 → max(0, 3) = 3
- Position [1,2]: 4 → max(0, 4) = 4
- Position [2,0]: 9 → max(0, 9) = 9
- Position [2,1]: 3 → max(0, 3) = 3
- Position [2,2]: 9 → max(0, 9) = 9

**After ReLU (3×3):**

```
[0  2  0]
[2  3  4]
[9  3  9]
```

---

### **Step 3: Apply 2×2 Max Pooling**

**Max Pooling:**

- Slide 2×2 window over input
- Take maximum value in each window
- Stride = 2 (default for pooling)

**Output Size after Pooling:**
$$H_{out} = \frac{H_{in} - K}{S} + 1 = \frac{3 - 2}{2} + 1 = 1.5$$

**Note:** Since 3×3 with 2×2 pooling doesn't divide evenly, we can only extract ONE 2×2 window from top-left, OR we need to specify stride=1 or use padding.

**Assuming stride=2 (standard), we get limited coverage. Let's use stride=1 for complete coverage:**

**Pool Window 1 (Top-Left 2×2):**

```
Input region:
[0  2]
[2  3]

Max = max(0, 2, 2, 3) = 3

Output[0,0] = 3
```

**Pool Window 2 (Top-Right 2×2):**

```
Input region:
[2  0]
[3  4]

Max = max(2, 0, 3, 4) = 4

Output[0,1] = 4
```

**Pool Window 3 (Bottom-Left 2×2):**

```
Input region:
[2  3]
[9  3]

Max = max(2, 3, 9, 3) = 9

Output[1,0] = 9
```

**Pool Window 4 (Bottom-Right 2×2):**

```
Input region:
[3  4]
[3  9]

Max = max(3, 4, 3, 9) = 9

Output[1,1] = 9
```

---

### **FINAL ANSWER:**

**Final Feature Map after Convolution + ReLU + MaxPooling (2×2):**

```
[3  4]
[9  9]
```

**Summary of Operations:**

1. **Convolution (5×5 → 3×3):** Applied 3×3 kernel to 5×5 input
2. **ReLU (3×3 → 3×3):** Removed negative value at position [0,2]
3. **MaxPooling (3×3 → 2×2):** Extracted maximum values from 2×2 windows

---

## **Question 2: CNN Parameter Calculation (5 Marks)**

### **Given:**

**Input Image:** 32×32×3 (RGB)

**Step 1 - Convolution Layer:**

- Kernel size: 3×3
- Number of filters: 5
- Padding: 1
- Stride: 2

**Step 2 - Max Pooling Layer:**

- Window size: 2×2
- Stride: 2

**Task:** Calculate number of parameters at each step

---

### **Step 1: Convolution Layer Parameters**

**Formula for Parameters in Conv Layer:**
$$\text{Parameters} = (K \times K \times C_{in} \times C_{out}) + C_{out}$$

Where:

- $K$ = Kernel size (3×3)
- $C_{in}$ = Input channels (3 for RGB)
- $C_{out}$ = Number of filters (5)
- $+C_{out}$ = Bias terms (one per filter)

**Calculation:**
$$\text{Parameters} = (3 \times 3 \times 3 \times 5) + 5$$
$$= (9 \times 3 \times 5) + 5$$
$$= (27 \times 5) + 5$$
$$= 135 + 5$$
$$= 140$$

**Breakdown:**

- **Weights:** 3×3×3×5 = 135
  - Each filter: 3×3×3 = 27 weights
  - 5 filters: 27×5 = 135 weights total
- **Biases:** 5 (one per filter)
- **Total:** 140 parameters

---

**Output Size after Conv Layer:**

**Formula:**
$$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$$

Where:

- $H_{in}$ = 32
- $K$ = 3
- $P$ = 1
- $S$ = 2

**Calculation:**
$$H_{out} = \frac{32 - 3 + 2(1)}{2} + 1$$
$$= \frac{32 - 3 + 2}{2} + 1$$
$$= \frac{31}{2} + 1$$
$$= 15.5 + 1$$
$$= 16.5$$

**Note:** Since we can't have 16.5, floor to 16.

**Output Dimensions:** 16×16×5

**Verification:**

- Width: $(32 - 3 + 2)/2 + 1 = 16$
- Height: $(32 - 3 + 2)/2 + 1 = 16$
- Depth: 5 filters

---

### **Step 2: Max Pooling Layer Parameters**

**Important Note:**
**Pooling layers have ZERO learnable parameters!**

Pooling only performs a fixed mathematical operation (max or average), no weights or biases to learn.

**Parameters in Max Pooling Layer: 0**

---

**Output Size after Max Pooling:**

**Formula:**
$$H_{out} = \frac{H_{in} - K}{S} + 1$$

Where:

- $H_{in}$ = 16 (from previous layer)
- $K$ = 2 (window size)
- $S$ = 2 (stride)

**Calculation:**
$$H_{out} = \frac{16 - 2}{2} + 1$$
$$= \frac{14}{2} + 1$$
$$= 7 + 1$$
$$= 8$$

**Output Dimensions:** 8×8×5

**Summary of pooling:**

- Width: $(16 - 2)/2 + 1 = 8$
- Height: $(16 - 2)/2 + 1 = 8$
- Depth: 5 (unchanged - pooling doesn't change number of channels)

---

### **FINAL ANSWER:**

**Step 1 (Convolution Layer):**

- **Parameters:** 140
  - Weights: 135
  - Biases: 5
- **Output Shape:** 16×16×5

**Step 2 (Max Pooling Layer):**

- **Parameters:** 0 (no learnable parameters)
- **Output Shape:** 8×8×5

**Total Parameters in CNN:** 140

---

### **Summary Table:**

| Layer       | Input Shape | Output Shape | Parameters | Calculation             |
| ----------- | ----------- | ------------ | ---------- | ----------------------- |
| **Input**   | 32×32×3     | -            | -          | -                       |
| **Conv1**   | 32×32×3     | 16×16×5      | **140**    | (3×3×3×5) + 5 = 135 + 5 |
| **MaxPool** | 16×16×5     | 8×8×5        | **0**      | No learnable parameters |
| **Total**   | -           | -            | **140**    | -                       |

---

## **Important Formulas (Note for Exam)**

### **1. Output Size Formula (Convolution/Pooling):**

$$H_{out} = \frac{H_{in} - K + 2P}{S} + 1$$

Where:

- $H_{in}$ = Input height/width
- $K$ = Kernel/window size
- $P$ = Padding (0 if not specified)
- $S$ = Stride (1 if not specified for conv, 2 for pooling)

### **2. Parameters in Convolution Layer:**

$$\text{Parameters} = (K \times K \times C_{in} \times C_{out}) + C_{out}$$

Where:

- $K$ = Kernel size
- $C_{in}$ = Input channels
- $C_{out}$ = Number of filters (output channels)
- $+C_{out}$ = Bias terms

### **3. ReLU Activation:**

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### **4. Max Pooling:**

- Takes maximum value from each window
- **No parameters** (fixed operation)
- Reduces spatial dimensions
- Typically uses stride = window size

---

## **Exam Tips:**

**For Convolution Problems:**

1. ✓ Calculate output size first using formula
2. ✓ Show all 9 positions systematically (for 3×3 output)
3. ✓ Element-wise multiply and sum for each position
4. ✓ Apply activation function (ReLU, Sigmoid, etc.)
5. ✓ Apply pooling if specified

**For Parameter Counting:**

1. ✓ Remember: Parameters = Weights + Biases
2. ✓ Conv layer: Use $(K \times K \times C_{in} \times C_{out}) + C_{out}$
3. ✓ Pooling layer: Always 0 parameters
4. ✓ Track output dimensions at each layer

**Common Mistakes to Avoid:**

- ✗ Forgetting to add biases in parameter count
- ✗ Counting pooling parameters (they're 0!)
- ✗ Not applying ReLU correctly (negatives become 0)
- ✗ Using wrong stride in output size formula

---

**END OF SOLUTIONS - Ready for Exam! 📝✅**
