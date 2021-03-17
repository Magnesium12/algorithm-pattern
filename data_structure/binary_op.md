# 二进制

## 常见二进制操作

### 基本操作

a=0^a=a^0

0=a^a

由上面两个推导出：a=a^b^b

### 交换两个数

a=a^b

b=a^b

a=a^b

### 移除最后一个 1

a=n&(n-1)

### 获取最后一个 1

diff=(n&(n-1))^n

## 常见题目

[single-number](https://leetcode-cn.com/problems/single-number/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans =0;
        for(auto i:nums){
            ans^=i;
        }
        return ans;
    }
};
```

[single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)

> 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

**解法1**：按位统计个数，然后模3

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans=0;
        for(int i=0;i<32;i++){
            int bit=0;
            for(auto num:nums){
                bit+= (num>>i)&1;
            }

            ans ^= (bit%3)<<i;
        }
        return ans;
    }
};
```
**解法2**：

和解法一类似于。构造有限状态机（类似于逻辑电路），00->01->10->00用来表示数模3的状态0，1，2，0.

>两位bit可以判断4种情况，由于数字最多出现3次，所以将11和00视为同一种状态

| next==1 | once  | twice | state |
| :-----: | :---: | :---: | :---- |
|    1    |   0   |   0   | 0     |
|    1    |   1   |   0   | 1     |
|    1    |   0   |   1   | 2     |

```cpp
class Solution
{
public:
    int singleNumber(vector<int> &nums)
    {
        int once = 0, twice = 0;
        for (auto num : nums)
        {
            once = ~twice & (num ^ once);
            twice = ~once & (num ^ twice);
        }
        return once;
    }
};
```


[single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)

> 给定一个整数数组  `nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

**解法**：a^b，找出最近的一个不同位。与操作将数组拆分为两部分

```cpp
class Solution
{
public:
    vector<int> singleNumber(vector<int> &nums)
    {
        int bit = 0;
        for (auto i : nums)
        {
            bit ^= i;
        }
        int div = 1;
        while (!(div & bit))
        {
            div <<= 1;
        }
        int a = 0, b = 0;
        for (auto i : nums)
        {
            if (i & div)
                a ^= i;
            else
                b ^= i;
        }
        return vector<int>{a, b};
    }
};

```

[number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)

> 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’  的个数（也被称为[汉明重量](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F)）。

```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int ans=0;
        while(n){
            n=n&(n-1);
            ans++;
        }
        return ans;
    }
};
```

[counting-bits](https://leetcode-cn.com/problems/counting-bits/)

> 给定一个非负整数  **num**。对于  0 ≤ i ≤ num  范围中的每个数字  i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

```cpp
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> dp(num+1,0);

        for(int i=1;i<=num;i++){
            dp[i]=dp[i&(i-1)]+1;
        }

        return dp;
    }
};
```


[reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)

> 颠倒给定的 32 位无符号整数的二进制位。

思路：依次颠倒即可

```go
func reverseBits(num uint32) uint32 {
    var res uint32
    var pow int=31
    for num!=0{
        // 把最后一位取出来，左移之后累加到结果中
        res+=(num&1)<<pow
        num>>=1
        pow--
    }
    return res
}
```
**解法2**：

分治策略，将32位分成较小的块，再将块反转即可
![Leetcode图例](https://pic.leetcode-cn.com/c57a82424197ba1f4091a67cc4a6c575b35dcc0bf9d077415838d3b22d4b1ff3-file_1585801736118)

```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        n = (n>>16)|(n<<16);
        n= ((n&0xff00ff00)>>8)|((n&0x00ff00ff)<<8);
        n= ((n&0xf0f0f0f0)>>4)|((n&0x0f0f0f0f)<<4);
        n= ((n&0xcccccccc)>>2)|((n&0x33333333)<<2);
        n= ((n&0xaaaaaaaa)>>1)|((n&0x55555555)<<1);
        return n;
    }
};
```

[bitwise-and-of-numbers-range](https://leetcode-cn.com/problems/bitwise-and-of-numbers-range/)

> 给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。

**思路**：找相同的前缀，且
1.数 n 比特位不能大于 m，否则没有前缀
2.找出从左往右 i位相同，则i+1及之后必然有相与为零，如1000&0101

```cpp
class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        while(right>left){
            right=right&(right-1);
        }
        return right;
    }
};
```

## 练习

- [ ] [single-number](https://leetcode-cn.com/problems/single-number/)
- [ ] [single-number-ii](https://leetcode-cn.com/problems/single-number-ii/)
- [ ] [single-number-iii](https://leetcode-cn.com/problems/single-number-iii/)
- [ ] [number-of-1-bits](https://leetcode-cn.com/problems/number-of-1-bits/)
- [ ] [counting-bits](https://leetcode-cn.com/problems/counting-bits/)
- [ ] [reverse-bits](https://leetcode-cn.com/problems/reverse-bits/)
