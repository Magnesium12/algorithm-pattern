# 动态规划

## 背景

先从一道题目开始~

如题  [triangle](https://leetcode-cn.com/problems/triangle/)

> 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

例如，给定三角形：

```text
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

自顶向下的最小路径和为  11（即，2 + 3 + 5 + 1 = 11）。

使用 DFS（遍历 或者 分治法）

遍历

![image.png](https://img.fuiboom.com/img/dp_triangle.png)

分治法

![image.png](https://img.fuiboom.com/img/dp_dc.png)

优化 DFS，缓存已经被计算的值（称为：记忆化搜索 本质上：动态规划）

![image.png](https://img.fuiboom.com/img/dp_memory_search.png)

动态规划就是把大问题变成小问题，并解决了小问题重复计算的方法称为动态规划

动态规划和 DFS 区别

- 二叉树 子问题是没有交集，所以大部分二叉树都用递归或者分治法，即 DFS，就可以解决
- 像 triangle 这种是有重复走的情况，**子问题是有交集**，所以可以用动态规划来解决

动态规划，自底向上

```go
func minimumTotal(triangle [][]int) int {
	if len(triangle) == 0 || len(triangle[0]) == 0 {
		return 0
	}
	// 1、状态定义：f[i][j] 表示从i,j出发，到达最后一层的最短路径
	var l = len(triangle)
	var f = make([][]int, l)
	// 2、初始化
	for i := 0; i < l; i++ {
		for j := 0; j < len(triangle[i]); j++ {
			if f[i] == nil {
				f[i] = make([]int, len(triangle[i]))
			}
			f[i][j] = triangle[i][j]
		}
	}
	// 3、递推求解
	for i := len(triangle) - 2; i >= 0; i-- {
		for j := 0; j < len(triangle[i]); j++ {
			f[i][j] = min(f[i+1][j], f[i+1][j+1]) + triangle[i][j]
		}
	}
	// 4、答案
	return f[0][0]
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

```

动态规划，自顶向下

```go
// 测试用例：
// [
// [2],
// [3,4],
// [6,5,7],
// [4,1,8,3]
// ]
func minimumTotal(triangle [][]int) int {
    if len(triangle) == 0 || len(triangle[0]) == 0 {
        return 0
    }
    // 1、状态定义：f[i][j] 表示从0,0出发，到达i,j的最短路径
    var l = len(triangle)
    var f = make([][]int, l)
    // 2、初始化
    for i := 0; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            if f[i] == nil {
                f[i] = make([]int, len(triangle[i]))
            }
            f[i][j] = triangle[i][j]
        }
    }
    // 递推求解
    for i := 1; i < l; i++ {
        for j := 0; j < len(triangle[i]); j++ {
            // 这里分为两种情况：
            // 1、上一层没有左边值
            // 2、上一层没有右边值
            if j-1 < 0 {
                f[i][j] = f[i-1][j] + triangle[i][j]
            } else if j >= len(f[i-1]) {
                f[i][j] = f[i-1][j-1] + triangle[i][j]
            } else {
                f[i][j] = min(f[i-1][j], f[i-1][j-1]) + triangle[i][j]
            }
        }
    }
    result := f[l-1][0]
    for i := 1; i < len(f[l-1]); i++ {
        result = min(result, f[l-1][i])
    }
    return result
}
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
```

## 递归和动规关系

递归是一种程序的实现方式：函数的自我调用

```go
Function(x) {
	...
	Funciton(x-1);
	...
}
```

动态规划：是一种解决问 题的思想，大规模问题的结果，是由小规模问 题的结果运算得来的。动态规划可用递归来实现(Memorization Search)

## 使用场景

满足两个条件

- 满足以下条件之一
  - 求最大/最小值（Maximum/Minimum ）
  - 求是否可行（Yes/No ）
  - 求可行个数（Count(\*) ）
- 满足不能排序或者交换（Can not sort / swap ）

如题：[longest-consecutive-sequence](https://leetcode-cn.com/problems/longest-consecutive-sequence/)  位置可以交换，所以不用动态规划

## 四点要素

1. 状态方程 State Function
   - 状态之间的联系，怎么通过小的状态，来算大的状态
2. 初始化 Intialization
   - 最极限的小状态是什么, 起点
3. 答案 Answer
   - 最大的那个状态是什么，终点

## 常见四种类型

1. Matrix DP (10%)
1. Sequence (40%)
1. Two Sequences DP (40%)
1. Backpack (10%)

> 注意点
>
> - 贪心算法大多题目靠背答案，所以如果能用动态规划就尽量用动规，不用贪心算法

## 1、矩阵类型（10%）

### [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)

> 给定一个包含非负整数的  *m* x *n*  网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

思路：动态规划
1、state function: function: f[x][y] = min(f[x-1][y], f[x][y-1]) + A[x][y]
3、intialize: 

$$
\begin{array}{l}
\bullet \text { 当 } i>0 \text { 且 } j=0 \text { 时, } d p[i][0]=d p[i-1][0]+\operatorname{grid}[i][0] \\
\bullet\text {  当 } i=0 \text { 且 } j>0 \text { 时, } d p[0][j]=d p[0][j-1]+g r i d[0][j] \\
\bullet \text { 当 } i>0 \text { 且 } j>0 \text { 时, } d p[i][j]=\min (d p[i-1][j], d p[i][j-1])+\operatorname{grid}[i][j]
\end{array}
$$


4、answer: f[n-1][m-1]

```go
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n =grid[0].size();

        // 优化后的dp
        vector<int> dp(n+1,INT_MAX/2);
        dp[1]=0;

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                dp[j+1]=min(dp[j],dp[j+1])+grid[i][j];
            }
        }

        return dp[n];
    }
};
```

### [unique-paths](https://leetcode-cn.com/problems/unique-paths/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？

```cpp
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<int> vec(n,1);
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                vec[j]=vec[j]+vec[j-1];
            }
        }

        return vec[n-1];
    }
};
```

### [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
> 问总共有多少条不同的路径？
> 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

```cpp
class Solution
{
public:
    int uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid)
    {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();

        vector<int> dp(n);
        dp[0] = (obstacleGrid[0][0] == 0);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (obstacleGrid[i][j] == 1)
                    dp[j] = 0;
                else if (j - 1 >= 0)
                    dp[j] = dp[j] + dp[j - 1];
            }
        }
        return dp[n - 1];
    }
};
```

## 2、序列类型（40%）

### [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)

> 假设你正在爬楼梯。需要  *n*  阶你才能到达楼顶。

```cpp
class Solution {
public:
    int climbStairs(int n) {
        if(n==1){
            return 1;
        }
        if(n==2){
            return 2;
        }

        vector<int> stair{1,2};

        for(int i=2;i<n;i++){
            swap(stair[0],stair[1]);
            stair[1]=stair[0]+stair[1];
        }
        return stair[1];
    }
};
```

### [jump-game](https://leetcode-cn.com/problems/jump-game/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 判断你是否能够到达最后一个位置。

```cpp
public:
    bool canJump(vector<int> &nums)
    {
        int right_most = 0;
        for (int i = 0; i < nums.size(); i++)
        {
            if (i > right_most)
                return false;
            right_most = max(right_most, i + nums[i]);
        }
        return true;
    }
};
```

### [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)

> 给定一个非负整数数组，你最初位于数组的第一个位置。
> 数组中的每个元素代表你在该位置可以跳跃的最大长度。
> 你的目标是使用最少的跳跃次数到达数组的最后一个位置。

**解法一**：贪心

```cpp
class Solution
{
public:
    int jump(vector<int> &nums)
    {
        int right_board = 0;
        int step = 0;
        for (int i = 0; i < nums.size();)
        {
            int tmp_rb = right_board;
            for (; i <= tmp_rb; i++)
            {
                right_board = max(i + nums[i], right_board);
            }
            step++;
            if (right_board >= nums.size() - 1)
                break;
        }
        return nums.size() == 1 ? 0 : step;
    }
};
```

### [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

> 给定一个字符串 _s_，将 _s_ 分割成一些子串，使每个子串都是回文串。
> 返回符合要求的最少分割次数。

**思路**：两次dp
- 根据回文串的最少分割次数，首先写出第一个状态转移方程，$f[i]$ 指到达 $i$ 处时，回文串分割所需的最少次数
  
$$
f[i]=\min _{0 \leq \jmath<i}\{f[j]\}+1, \quad \text { 其中 } s[j+1 . . i] \text { 是一个回文串 }
$$

- 如何确定 $s[i,j]$ 是回文串，再写出第二个状态转移方程

$$
g(i, j)=\left\{\begin{array}{ll}
\text { True, } & i \geq j \\
g(i+1, j-1) \wedge(s[i]=s[j]), & \text { otherwise }
\end{array}\right.
$$

> 当然实际上我们是先利用一个[i,j]的二维数组来维护从i到j的回文串真伪性，然后再利用各部分回文串来动态规划确定终点的最少次数
```cpp
class Solution
{
public:
    int minCut(string s)
    {
        int n = s.size();
        vector<vector<bool>> sig(n, vector<bool>(n, true));

        for (int i = n - 1; i >= 0; i--)
        {
            for (int j = i + 1; j < n; j++)
            {
                // 注意到sig[i,j] 需要sig[i+1,j-1]，所以i必须从高向低遍历，j则需要从i+1向高位遍历
                sig[i][j] = (s[i] == s[j]) && (sig[i + 1][j - 1]);
            }
        }

        vector<int> v(n, INT_MAX);

        for (int i = 0; i < n; i++)
        {
            if (sig[0][i])
            {
                v[i] = 0;
            }
            else
            {
                for (int j = 0; j < i; j++)
                {
                    if (sig[j + 1][i])
                        v[i] = min(v[i], v[j] + 1);
                }
            }
        }
        return v[n - 1];
    }
};
```

注意点

- 判断回文字符串时，可以提前用动态规划算好，减少时间复杂度

### [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

> 给定一个无序的整数数组，找到其中最长上升子序列的长度。

**思路：动态规划**

$f(i)$ 表示遍历到 `nums[i]` 处的最大递增子序列长度
$$
f(i)=\left\{\begin{array}{ll}
max(f(j))+1 & j \in [0, i),nums[i]>nums[j]\\
1, & \text { otherwise }
\end{array}\right.
$$
边界条件：$f(0)=1$
```cpp
class Solution
{
public:
    int lengthOfLIS(vector<int> &nums)
    {
        int len = nums.size();
        vector<int> dp(len, 1);
        int max_len = 1;
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (nums[j] < nums[i])
                    dp[i] = max(dp[j] + 1, dp[i]);
            }
            max_len = max(dp[i], max_len);
        }
        return max_len;
    }
};
```

时间复杂度 $O(n^2)$

**思路二：贪心算法+二分查找**

考虑一个简单的贪心，如果我们要使上升子序列尽可能的长，则我们需要让序列上升得尽可能慢，因此我们希望每次在上升子序列最后加上的那个数尽可能的小。

维护一个 greed 数组，`greed[i]`指i+1长度的最长递增子序列末尾元素的最小值，遍历 nums 更新 greed ，最终`greed.size()` 即为最长递增子序列。

```cpp
class Solution
{
public:
    int lengthOfLIS(vector<int> &nums)
    {
        int len = nums.size();
        vector<int> greed;
        greed.emplace_back(nums[0]);
        for (int i = 0; i < len; i++)
        {
            if (nums[i] > greed.back())
                greed.emplace_back(nums[i]);
            else
            {
                int l = 0, r = greed.size(), pos = 0;
                while (l <= r)
                {
                    int mid = (l + r) >> 1;
                    if (greed[mid] >= nums[i])
                    {
                        r = mid - 1;
                        pos = mid;
                    }
                    else
                        l = mid + 1;
                }
                greed[pos] = nums[i];
            }
        }
        return greed.size();
    }
};
```

### [word-break](https://leetcode-cn.com/problems/word-break/)

> 给定一个**非空**字符串  *s*  和一个包含**非空**单词列表的字典  *wordDict*，判定  *s*  是否可以被空格拆分为一个或多个在字典中出现的单词。

```cpp
class Solution
{
public:
    bool wordBreak(string s, vector<string> &wordDict)
    {
        int n = s.size();
        vector<int> dp(n + 1);
        dp[0] = 1;
        for (int i = 0; i < n; i++)
        {
            if (dp[i] == 1)
            {
                for (int j = 0; j < wordDict.size(); j++)
                {
                    int word_len = wordDict[j].size();
                    if (i + word_len <= n && s.substr(i, word_len) == wordDict[j])
                    {
                        dp[i + word_len] = 1;
                    }
                }
            }
        }
        for (auto i : dp)
        {
            cout << i;
        }
        return dp[n] == 1;
    }
};

```

小结

常见处理方式是给 0 位置占位，这样处理问题时一视同仁，初始化则在原来基础上 length+1，返回结果 f[n]

- 状态可以为前 i 个
- 初始化 length+1
- 取值 index=i-1
- 返回值：f[n]或者 f[m][n]

## Two Sequences DP（40%）

### [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)

> 给定两个字符串  text1 和  text2，返回这两个字符串的最长公共子序列。
> 一个字符串的   子序列   是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
> 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

```go
func longestCommonSubsequence(a string, b string) int {
    // dp[i][j] a前i个和b前j个字符最长公共子序列
    // dp[m+1][n+1]
    //   ' a d c e
    // ' 0 0 0 0 0
    // a 0 1 1 1 1
    // c 0 1 1 2 1
    //
    dp:=make([][]int,len(a)+1)
    for i:=0;i<=len(a);i++ {
        dp[i]=make([]int,len(b)+1)
    }
    for i:=1;i<=len(a);i++ {
        for j:=1;j<=len(b);j++ {
            // 相等取左上元素+1，否则取左或上的较大值
            if a[i-1]==b[j-1] {
                dp[i][j]=dp[i-1][j-1]+1
            } else {
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
            }
        }
    }
    return dp[len(a)][len(b)]
}
func max(a,b int)int {
    if a>b{
        return a
    }
    return b
}
```

注意点

- go 切片初始化

```go
dp:=make([][]int,len(a)+1)
for i:=0;i<=len(a);i++ {
    dp[i]=make([]int,len(b)+1)
}
```

- 从 1 开始遍历到最大长度
- 索引需要减一

### [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

> 给你两个单词  word1 和  word2，请你计算出将  word1  转换成  word2 所使用的最少操作数  
> 你可以对一个单词进行如下三种操作：
> 插入一个字符
> 删除一个字符
> 替换一个字符

思路：和上题很类似，相等则不需要操作，否则取删除、插入、替换最小操作次数的值+1

```go
func minDistance(word1 string, word2 string) int {
    // dp[i][j] 表示a字符串的前i个字符编辑为b字符串的前j个字符最少需要多少次操作
    // dp[i][j] = OR(dp[i-1][j-1]，a[i]==b[j],min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1)
    dp:=make([][]int,len(word1)+1)
    for i:=0;i<len(dp);i++{
        dp[i]=make([]int,len(word2)+1)
    }
    for i:=0;i<len(dp);i++{
        dp[i][0]=i
    }
    for j:=0;j<len(dp[0]);j++{
        dp[0][j]=j
    }
    for i:=1;i<=len(word1);i++{
        for j:=1;j<=len(word2);j++{
            // 相等则不需要操作
            if word1[i-1]==word2[j-1] {
                dp[i][j]=dp[i-1][j-1]
            }else{ // 否则取删除、插入、替换最小操作次数的值+1
                dp[i][j]=min(min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1
            }
        }
    }
    return dp[len(word1)][len(word2)]
}
func min(a,b int)int{
    if a>b{
        return b
    }
    return a
}
```

说明

> 另外一种做法：MAXLEN(a,b)-LCS(a,b)

## 零钱和背包（10%）

### [coin-change](https://leetcode-cn.com/problems/coin-change/)

> 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回  -1。

思路：和其他 DP 不太一样，i 表示钱或者容量

```go
func coinChange(coins []int, amount int) int {
    // 状态 dp[i]表示金额为i时，组成的最小硬币个数
    // 推导 dp[i]  = min(dp[i-1], dp[i-2], dp[i-5])+1, 前提 i-coins[j] > 0
    // 初始化为最大值 dp[i]=amount+1
    // 返回值 dp[n] or dp[n]>amount =>-1
    dp:=make([]int,amount+1)
    for i:=0;i<=amount;i++{
        dp[i]=amount+1
    }
    dp[0]=0
    for i:=1;i<=amount;i++{
        for j:=0;j<len(coins);j++{
            if  i-coins[j]>=0  {
                dp[i]=min(dp[i],dp[i-coins[j]]+1)
            }
        }
    }
    if dp[amount] > amount {
        return -1
    }
    return dp[amount]

}
func min(a,b int)int{
    if a>b{
        return b
    }
    return a
}
```

注意

> dp[i-a[j]] 决策 a[j]是否参与

### [backpack](https://www.lintcode.com/problem/backpack/description)

> 在 n 个物品中挑选若干物品装入背包，最多能装多满？假设背包的大小为 m，每个物品的大小为 A[i]

```go
func backPack (m int, A []int) int {
    // write your code here
    // f[i][j] 前i个物品，是否能装j
    // f[i][j] =f[i-1][j] f[i-1][j-a[i] j>a[i]
    // f[0][0]=true f[...][0]=true
    // f[n][X]
    f:=make([][]bool,len(A)+1)
    for i:=0;i<=len(A);i++{
        f[i]=make([]bool,m+1)
    }
    f[0][0]=true
    for i:=1;i<=len(A);i++{
        for j:=0;j<=m;j++{
            f[i][j]=f[i-1][j]
            if j-A[i-1]>=0 && f[i-1][j-A[i-1]]{
                f[i][j]=true
            }
        }
    }
    for i:=m;i>=0;i--{
        if f[len(A)][i] {
            return i
        }
    }
    return 0
}

```

### [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)

> 有 `n` 个物品和一个大小为 `m` 的背包. 给定数组 `A` 表示每个物品的大小和数组 `V` 表示每个物品的价值.
> 问最多能装入背包的总价值是多大?

思路：f[i][j] 前 i 个物品，装入 j 背包 最大价值

```go
func backPackII (m int, A []int, V []int) int {
    // write your code here
    // f[i][j] 前i个物品，装入j背包 最大价值
    // f[i][j] =max(f[i-1][j] ,f[i-1][j-A[i]]+V[i]) 是否加入A[i]物品
    // f[0][0]=0 f[0][...]=0 f[...][0]=0
    f:=make([][]int,len(A)+1)
    for i:=0;i<len(A)+1;i++{
        f[i]=make([]int,m+1)
    }
    for i:=1;i<=len(A);i++{
        for j:=0;j<=m;j++{
            f[i][j]=f[i-1][j]
            if j-A[i-1] >= 0{
                f[i][j]=max(f[i-1][j],f[i-1][j-A[i-1]]+V[i-1])
            }
        }
    }
    return f[len(A)][m]
}
func max(a,b int)int{
    if a>b{
        return a
    }
    return b
}
```

## 练习

Matrix DP (10%)

- [ ] [triangle](https://leetcode-cn.com/problems/triangle/)
- [ ] [minimum-path-sum](https://leetcode-cn.com/problems/minimum-path-sum/)
- [ ] [unique-paths](https://leetcode-cn.com/problems/unique-paths/)
- [ ] [unique-paths-ii](https://leetcode-cn.com/problems/unique-paths-ii/)

Sequence (40%)

- [ ] [climbing-stairs](https://leetcode-cn.com/problems/climbing-stairs/)
- [ ] [jump-game](https://leetcode-cn.com/problems/jump-game/)
- [ ] [jump-game-ii](https://leetcode-cn.com/problems/jump-game-ii/)
- [ ] [palindrome-partitioning-ii](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)
- [ ] [longest-increasing-subsequence](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
- [ ] [word-break](https://leetcode-cn.com/problems/word-break/)

Two Sequences DP (40%)

- [ ] [longest-common-subsequence](https://leetcode-cn.com/problems/longest-common-subsequence/)
- [ ] [edit-distance](https://leetcode-cn.com/problems/edit-distance/)

Backpack & Coin Change (10%)

- [ ] [coin-change](https://leetcode-cn.com/problems/coin-change/)
- [ ] [backpack](https://www.lintcode.com/problem/backpack/description)
- [ ] [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)


- 猿辅导：击鼓传花
```cpp
const int MOD = 10000007;

// 矩阵快速幂(方阵)
vector<vector<int>> matMul(vector<vector<int>> &A, vector<vector<int>> &B)
{
    vector<vector<int>> ret(A.size(), vector<int>(B[0].size(), 0));
    for (size_t i = 0; i < A.size(); i++)
    {
        for (size_t j = 0; j < B[0].size(); j++)
        {
            for (size_t k = 0; k < A[0].size(); k++)
            {
                ret[i][j] += A[i][k] * B[k][j];
                ret[i][j] %= MOD;
            }
        }
    }
    return ret;
}

vector<vector<int>> matQuickPow(vector<vector<int>> &mat, int n)
{
    vector<vector<int>> ret = {{1, 0}, {0, 1}};
    while (n)
    {
        if (n & 1)
            ret = matMul(ret, mat);

        mat = matMul(mat, mat);
        n >>= 1;
    }
    return ret;
}

int main(int argc, char const *argv[])
{
    int N, K;
    cin >> N >> K;

    vector<vector<int>> mat = {{K - 2, K - 1}, {1, 0}};

    mat = matQuickPow(mat, N-2);
    for (auto i : mat)
    {
        for (auto j : i)
        {
            cout << j<<" ";
        }
        cout<<endl;
    }

    int ans = 0;
    vector<int> X = {K - 1, 0};
    ans = mat[0][0] * X[0] % MOD + mat[0][1] * X[1] % MOD;
    cout<<endl<<ans;
    return 0;
}

```