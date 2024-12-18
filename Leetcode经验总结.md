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

### 数据结构思路

- **二叉搜索树** → 是否使用中序遍历为递增数列的特点

- **动态维护Top K** → 堆

### 常见方法模板

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

##### 二叉树层序遍历

```python
# 含深度的二叉树层序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        queue = deque([(root, 0)])
        max_depth = -1
        tree_at_depth = dict()
        
        while(queue):
            node, depth = queue.popleft()

            if node:
                if depth > max_depth:
                    max_depth = depth
                    tree_at_depth[depth] = [node.val]
                else:
                    tree_at_depth[depth].append(node.val)

                queue.append([node.left, depth + 1])
                queue.append([node.right, depth + 1])
        return [tree_at_depth[i] for i in range(max_depth + 1)]
```





#### 动态规划思考路径

1. 是否可以拆解为 f(k+1) = g(f(k)) 或者 g(f(1, ..., k)) 的逻辑
2. 写出状态转移公式/代码逻辑

具体问题具体分析！

#### 滑动窗口思考路径

如何移动起始位置

1. 找到符合条件的区间，移动起始位置收缩区间

#### KMP模板

思路：通过前缀表记录，在匹配失败时从下一个合适位置继续检索，减少重复匹配，将时间复杂度从O(m * n) 降低到 O(m + n)

理解：【帮你把KMP算法学个通透！（理论篇）】 https://www.bilibili.com/video/BV1PD4y1o7nd/?share_source=copy_web&vd_source=5d08564ce088a80b77e6aa6282030c68

```python
# Leetcode 28
// 方法一
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        a=len(needle)
        b=len(haystack)
        if a==0:
            return 0
        next=self.getnext(a,needle)
        p=-1
        for j in range(b):
            while p>=0 and needle[p+1]!=haystack[j]:
                p=next[p]
            if needle[p+1]==haystack[j]:
                p+=1
            if p==a-1:
                return j-a+1
        return -1

    def getnext(self,a,needle):
        next=['' for i in range(a)]
        k=-1
        next[0]=k
        for i in range(1,len(needle)):
            while (k>-1 and needle[k+1]!=needle[i]):
                k=next[k]
            if needle[k+1]==needle[i]:
                k+=1
            next[i]=k
        return next

作者：代码随想录
链接：https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/solutions/732461/dai-ma-sui-xiang-lu-kmpsuan-fa-xiang-jie-mfbs/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```







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

## Python常见数据结构调用语法

### 列表（List）

- **特点**: 有序、可变、支持重复元素。
- **常用方法**: `append`, `insert`, `remove`, `pop`, `sort`, `reverse` 等。

```python
# 创建列表
my_list = [1, 2, 3, 4]

# 添加元素
my_list.append(5)       # [1, 2, 3, 4, 5]

# 删除元素
my_list.remove(99)      # [1, 2, 3, 4, 5]

# 获取最后一个元素并删除
last_element = my_list.pop()  # [1, 2, 3, 4], last_element = 5

# 插入最前 无直接实现函数，但可以：
my_list = [0] + my_list

# 创建固定长度列表
my_list = [0] * n

# 排序
my_list.sort()          # 升序排序

# 倒序
my_list.reverse()       # [4, 3, 2, 1]
```

### 集合（Set）

- **特点**: 无序、元素唯一、不支持重复。
- **常用方法**: `add`, `remove`, `union`, `intersection`, `difference` 等。

```python
# 创建集合
my_set = {1, 2, 3, 4}
# 添加元素 注意是add！
my_set.add(5)           # {1, 2, 3, 4, 5}
# 删除元素
my_set.remove(3)        # {1, 2, 4, 5}
```

### 双端队列（Deque）

- **特点**: 双端队列，可以从两端添加或移除元素。
- **需要导入模块**: `collections.deque`
- **常用方法**: `append`, `appendleft`, `pop`, `popleft`, `extend`, `extendleft` 等。

```python
from collections import deque

# 创建双端队列
my_deque = deque([1, 2, 3])

# 从右端添加元素
my_deque.append(4)      # deque([1, 2, 3, 4])

# 从左端添加元素
my_deque.appendleft(0)  # deque([0, 1, 2, 3, 4])

# 从右端移除元素
my_deque.pop()          # deque([0, 1, 2, 3])

# 从左端移除元素
my_deque.popleft()      # deque([1, 2, 3])
```

### 字典（Dictionary）

- **特点**: 键值对形式存储，无序（Python 3.7+ 的实现是插入顺序）。
- **常用方法**: `get`, `keys`, `values`, `items`, `pop` 等。

```python
# 创建字典
my_dict = {"a": 1, "b": 2, "c": 3}
# 添加/更新键值对
my_dict["d"] = 4        # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# 获取值
value = my_dict["a"]    # 1
# 删除键值对
my_dict.pop("b")        # {'a': 1, 'c': 3, 'd': 4}
del my_dict["b"]
# 遍历键值对
for key, value in my_dict.items():
    print(key, value)
# 打印值
print(my_dict.values())
```

### 堆栈（Stack）

- **特点**: 使用 `list` 模拟，遵循后进先出（LIFO）原则。

```python
stack = []
# 压栈
stack.append(1)
stack.append(2)         # [1, 2]
# 出栈
top = stack.pop()       # [1], top = 2
```

### 堆（Heap）

- 特点
  - 堆是一种完全二叉树，常用来快速找到最大值或最小值。
  - Python 的 `heapq` 模块提供最小堆实现，最大堆需要通过**对元素取负数实现**。
- **常用方法**: `heappush`, `heappop`, `heapify` 等。

```python
import heapq

# 最小堆
min_heap = []
heapq.heappush(min_heap, 3)   # [3]
heapq.heappush(min_heap, 1)   # [1, 3]
heapq.heappush(min_heap, 4)   # [1, 3, 4]
print(heapq.heappop(min_heap))  # 1, 堆变为 [3, 4]

# 最大堆（通过负值实现）
max_heap = []
heapq.heappush(max_heap, -3)  # [-3]
heapq.heappush(max_heap, -1)  # [-3, -1]
heapq.heappush(max_heap, -4)  # [-4, -1, -3]
print(-heapq.heappop(max_heap))  # 3, 堆变为 [-3, -1]

# 堆化
nums = [4, 1, 7, 3, 8]
heapq.heapify(nums)           # 最小堆 [1, 3, 7, 4, 8]
```

### 优先队列（Priority Queue）

- **特点**: 按照优先级处理元素，可以看作堆的高级应用。
- **实现方式**: 使用 `heapq` 模拟或 `queue.PriorityQueue`。

```python
import heapq

# 优先队列（使用元组存储优先级）
pq = []
heapq.heappush(pq, (2, "task2"))  # 优先级为 2 的任务
heapq.heappush(pq, (1, "task1"))  # 优先级为 1 的任务
heapq.heappush(pq, (3, "task3"))  # 优先级为 3 的任务
print(heapq.heappop(pq))          # (1, "task1")
```

