# Leetcode经验总结

[toc]

## General Guidelines

### 关键词思路

- **a涵盖b** → Counter(a) >= Counter(b)

- **遍历** → DFS/BFS

- **数组 + 考虑某元素两边** → 从左到右遍历 + 从右到左遍历
  - 238、135、42


### 方法思路

- **单调递增** → 二分查找  → 左闭右开/左闭右闭
- **单调递增/子问题单调递增** → 双指针  
- **排列/组合** → 回溯  
- **可将问题拆解为 总问题 = 前k问题 + 子问题** → 动态规划
- **求最大连续子串长度** → 滑动窗口
- **字符串匹配** → KMP
- **具有最大和的连续子数组** → Kadane 算法

### 数据结构

- **二叉搜索树** → 中序遍历

#### 回溯模板

```python
# Leetcode 77： 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 回溯递归helper函数
        def backtrack(s: int):
            if len(combination) == k:
                # 树到达叶子节点，列为解答
                combinations.append(combination.copy()) # 注意需要是copy()
                return
            for num in range(s, n + 1):
                # 枚举所有情况
                combination.append(num)
                backtrack(num + 1) # 顺着树往下走
                combination.pop() # 复原成前一种情况，即回溯操作
            
        combinations = list()
        combination = list()
        backtrack(1)
        return combinations
```

#### DFS模板

DFS（深度优先搜索，Depth First Search）是一种用来遍历或搜索树和图的算法。它按照“深入到底，再回退”的策略工作，优先访问每个节点的子节点，直到走到“最深”的路径，然后再回退并访问其他路径。

```python
# 递归实现DFS
def dfs(node, visited):
    if not node or node in visited:
        return
    visited.add(node)  # 标记当前节点为已访问
    print(node)  # 访问当前节点
    for neighbor in node.neighbors:  # 遍历邻居节点
        dfs(neighbor, visited)
# 栈实现DFS
def dfs_stack(start):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)  # 标记为已访问
            print(node)  # 访问当前节点
            stack.extend(node.neighbors)  # 将邻居节点加入栈
```

#### BFS模板

BFS（广度优先搜索，Breadth First Search）是一种用来遍历或搜索树和图的算法。它按照“从近到远，逐层遍历”的策略工作，先访问所有距离起点最近的节点，然后逐渐扩展到更远的节点。

```python
from collections import deque

def bfs(start):
    queue = deque([start])  # 初始化队列，放入起点
    visited = set()  # 用于记录访问过的节点
    while queue:
        node = queue.popleft()  # 从队列中取出一个节点
        if node not in visited:
            print(node)  # 访问当前节点
            visited.add(node)  # 标记为已访问
            for neighbor in node.neighbors:  # 将未访问的邻居加入队列
                if neighbor not in visited:
                    queue.append(neighbor)
```

#### 动态规划思考路径

1. 是否可以拆解为 f(k+1) = g(f(k)) 或者 g(f(1, ..., k)) 的逻辑
2. 写出状态转移公式

#### 滑动窗口思考路径

如何移动起始位置

1. 找到符合条件的区间，移动起始位置收缩区间

#### KMP模板

思路：通过前缀表记录，减少重复匹配，将时间复杂度从O(m * n) 降低到 O(m + n)





#### Kadene算法

**动态规划** 的思想：

1. **问题分解**：

   - 每个位置的最大子数组和，只跟前一个位置的最大子数组和有关。
   - 换句话说：
     - 要么将当前元素加入之前的子数组中（延续当前子数组）。
     - 要么以当前元素作为新的子数组的起点（放弃之前的子数组）。

2. **状态转移公式**：

   - 假设 `max_ending_here` 是以当前元素结尾的子数组的最大和，

   - 那么：

     $max\_ending\_here = \max(nums[i], max\_ending\_here + nums[i])$

     - 如果把当前元素加入之前的子数组会更大，那就延续。
     - 否则，以当前元素重新开始新的子数组。

3. **全局最大值**：

   - 在计算 `max_ending_here` 的同时，记录一个全局的最大值 `max_so_far`。
   - 即： $max\_so\_far = \max(max\_so\_far, max\_ending\_here)$

#### 二叉树中序遍历

```python
p = root
st = []  # 用列表模拟实现栈的功能
while p is not None or st:  # 当节点 p 存在或栈非空时循环
    while p is not None:  # 不断深入左子树
        st.append(p)  # 当前节点入栈
        p = p.left  # 移动到左子节点
    p = st.pop()  # 栈顶元素出栈（回溯到上一个未访问的节点）
    proc(p.val)  # 处理当前节点的值
    p = p.right  # 转向当前节点的右子树
```



## 具体题目

### 数组/字符串

```python
# 买卖股票最佳时间
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        size = len(prices)
        dp0 = 0
        dp1 = -prices[0]
        for i in range (1, size):
            newDp0 = max(dp0, dp1 + prices[i])
            newDp1 = max(dp1, dp0 - prices[i])
            dp0 = newDp0
            dp1 = newDp1
        return dp0
```





### 双指针

解答核心：

1. 是否使用排序？
2. 是否需要简化/Reframe问题？
3. 左指针移动逻辑
4. 右指针移动逻辑



```python
# 三数之和
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # 1. sort
        nums.sort()
        # [-4, -1, -1, 0, 1, 2]
        n = len(nums)
        ans = list()

        # 2. use pointers
        for first in range(n):
            target = -nums[first]
            third = n - 1
            if first > 0 and nums[first - 1] == nums[first]:
                continue
            for second in range(first + 1, n - 1):
                if second > first + 1 and nums[second - 1] == nums[second]:
                    continue
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])
        return ans
```





### 滑动窗口

解答核心：

1. 左指针移动逻辑

2. 右指针移动逻辑

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        occ = set()
        right = -1
        ans = 0
        for i in range(n):
            if i != 0:
                occ.remove(s[i - 1])
            while right + 1 < n and s[right+1] not in occ:
                occ.add(s[right + 1])
                right += 1
            ans = max(ans, right - i + 1)
        return ans
```





### 矩阵





### 哈希表



#### 栈

##### 基本计算器

```python
class Solution:
    def calculate(self, s: str) -> int:
        def helper(index: int) -> (int, int):
            stack = []  # 栈保存当前括号内的结果
            num = 0  # 当前数字
            sign = '+'  # 当前符号
            i = index  # 当前字符索引

            while i < len(s):
                char = s[i]

                if char.isdigit():
                    # 累积当前数字
                    num = num * 10 + int(char)

                if char in "+-*/" or char == ')' or i == len(s) - 1:
                    # 处理前一个符号的运算逻辑
                    if sign == '+':
                        stack.append(num)
                    elif sign == '-':
                        stack.append(-num)
                    elif sign == '*':
                        stack.append(stack.pop() * num)
                    elif sign == '/':
                        # Python 特殊的整数除法处理
                        stack.append(int(stack.pop() / num))

                    # 更新符号并重置数字
                    sign = char
                    num = 0

                    # 如果遇到右括号，结束当前括号的计算
                    if char == ')':
                        return sum(stack), i

                elif char == '(':
                    # 遇到左括号，递归计算括号内的结果
                    num, i = helper(i + 1)

                i += 1

            return sum(stack), i

        # 去掉空格并从索引 0 开始处理
        s = s.replace(' ', '')
        result, _ = helper(0)
        return result
```

