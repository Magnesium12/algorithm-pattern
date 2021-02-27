# 栈和队列

## 简介

栈的特点是后入先出

![image.png](https://img.fuiboom.com/img/stack.png)

根据这个特点可以临时保存一些数据，之后用到依次再弹出来，常用于 DFS 深度搜索

队列一般常用于 BFS 广度搜索，类似一层一层的搜索

## Stack 栈

[min-stack](https://leetcode-cn.com/problems/min-stack/)

> 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

思路：

1. 栈存放差值，每次pop/top 元素时，根据差值和最小元素弹出数据；并利用差值更新min

2. 也可以用两个栈实现，一个最小栈始终保证最小值在顶部


```cpp
//思路一
class MinStack {
public:
    /** initialize your data structure here. */
    stack<int> stk;
    int min;
    MinStack() {
    }
    
    void push(int x) {
        if(stk.empty()){
            stk.push(x);
            min = x;
        }
        else{
            if(x-min>0){
                stk.push(x);
            }
            else{
                stk.push(x-min);
                 min = x;
            }
        }
        
    }
    
    void pop() {
        if(stk.top()<0){
            min=min-stk.top();
        }
        stk.pop();
    }
    
    int top() {
        return stk.top();
    }
    
    int getMin() {
        return min;
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
```

[evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

> **波兰表达式计算** > **输入:** `["2", "1", "+", "3", "*"]` > **输出:** 9
>
> **解释:** `((2 + 1) * 3) = 9`

思路：数字压入，运算符弹出

```cpp
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<int> number;

        int first,second;

        for(auto s:tokens){
             if( s == "+" || s == "-" || s == "*" || s == "/" )
            {                            
                int second  = number.top(); number.pop();
                int first   = number.top(); number.pop();
                if( s == "+" ){ number.push( first + second ); }
                if( s == "-" ){ number.push( first - second ); }
                if( s == "*" ){ number.push( first * second ); }
                if( s == "/" ){ number.push( first / second ); }
            }
            else
            {
                number.push(stoi(s));
            }
        }
        return number.top();
    }
};
```

[decode-string](https://leetcode-cn.com/problems/decode-string/)

> 给定一个经过编码的字符串，返回它解码后的字符串。
> s = "3[a]2[bc]", 返回 "aaabcbc".
> s = "3[a2[c]]", 返回 "accaccacc".
> s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".

思路：递归操作

```cpp
class Solution {
public:
    string recursion(string s,int &idx){
        string res;

        string tmp;
        int mul=0;
        while(idx!=s.size()){
            if(isdigit(s[idx])){
                mul=mul*10+(s[idx]-'0');
            }
            else if(s[idx]=='['){
                tmp=recursion(s,++idx);
                
                while(mul--){
                    
                    res+=tmp;
                }
    
                mul=0;
            }
            else if(isalpha(s[idx])){
                res+=s[idx];
            }
            else if(s[idx]==']'){
                break;
            }
            idx++;
        }
        return res;
    }
    string decodeString(string s) {
        int idx=0;
        return recursion(s,idx);
    }
};
```

- 利用栈操作

```cpp
class Solution {
public:
   string getDigit(string &s,int &p){
       string ret;
       while(isdigit(s[p])){
           ret+=s[p++];
       }
       return ret;
   }
   string getString(vector<string> &s){
       string ret;
       for(const auto& p:s){
           ret+=p;
       }
       return ret;
   }
    string decodeString(string s) {

        //模拟栈
        vector<string> vec;
        int i=0;
        while(i<s.size()){
            if(isdigit(s[i])){
                string num = getDigit(s,i);
                vec.push_back(num);
            }else if(isalpha(s[i])||s[i]=='['){
                vec.push_back(string(1,s[i++]));
            }else if(s[i]==']'){
                i++;
                vector<string> sub;
                while(vec.back()!="["){
                    sub.push_back(vec.back());
                    vec.pop_back();
                }
                reverse(sub.begin(),sub.end());

                //此时栈顶为'[',出栈
                vec.pop_back();

                //此时为重复次数数字,取出并出栈
                int repTime = stoi(vec.back());
                vec.pop_back();
                string ret,subString=getString(sub);
                while(repTime--){
                    ret+=subString;
                }
                vec.push_back(ret);
            }
        }
 
        return getString(vec);

    }
};
```

[binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

> 给定一个二叉树，返回它的*中序*遍历。

```cpp
// 思路：通过stack 保存已经访问的元素，用于原路返回
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        TreeNode* p=root;
        stack<TreeNode*> stk;
        vector<int> ans;

        //stk存放未被访问的父节点，p为当前节点
        while(!stk.empty()||p){
            if(p){
                //一路向左
                stk.push(p);
                p=p->left;
            }
            else{
                //左孩子为空则转向父节点
                p=stk.top();
                stk.pop();
                ans.push_back(p->val);
                p=p->right;
            }
        }
        return ans;

    }
};
```

复习一下递归遍历操作
```cpp

class Solution {
public:
    vector<int> ans;
    void dfs(TreeNode* root){
        if(!root){
            return;
        }
        dfs(root->left);
        ans.push_back(root->val);
        dfs(root->right);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        dfs(root);
        return ans;
    }
};
```

[clone-graph](https://leetcode-cn.com/problems/clone-graph/)

> 给你无向连通图中一个节点的引用，请你返回该图的深拷贝（克隆）。

思路一：DFS递归搜索 + hashmap标识访问

```cpp
class Solution {
public:
    unordered_map<int,Node*> hash;

    Node* cloneGraph(Node* node) {
        if(!node){
            return nullptr;
        }
        
        if(hash.find(node->val)!=hash.end()){
            return hash[node->val];
        }

        Node* nodeClone=new Node(node->val);
        hash[node->val]=nodeClone;

        for(auto &neighbor:node->neighbors){
            nodeClone->neighbors.emplace_back(cloneGraph(neighbor));
        }


        return nodeClone;
    }
};
```
思路二：BFS
```cpp
class Solution {
public:
    unordered_map<int,Node*> hash;

    Node* cloneGraph(Node* node) {
        if(!node){
            return nullptr;
        }

        unordered_map <Node*,Node*> hash;

        // 辅助队列：存放遍历层次中未遍历的节点
        queue<Node*> qu;

        qu.push(node);

        //标记已遍历节点
        hash[node]=new Node(node->val);

        //广度有限搜索
        while(!qu.empty()){
            Node* cur = qu.front();
            qu.pop();

            for(auto neighbor:cur->neighbors){
                if(hash.find(neighbor)==hash.end()){
                    //如果邻居初次访问，则存入hash中进行标记
                    hash[neighbor]=new Node(neighbor->val);
                    qu.push(neighbor);
                }
                //更新新节点的邻居列表
                hash[cur]->neighbors.emplace_back(hash[neighbor]);
            }

        }
        return hash[node];
    }
};
```
[number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)

> 给定一个由  '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

思路：通过深度搜索遍历可能性（注意标记已访问元素）

```cpp
class Solution {
public:
    int dfs(vector<vector<char>>& grid,int i,int j){
        //判断（i，j）是否需要遍历
        if(i>=grid.size()||i<0||j>=grid[0].size()||j<0||grid[i][j]=='0'){
            return 0;
        }
        if(grid[i][j]=='1'){
            //标记点
            grid[i][j]='0';
            return dfs(grid,i+1,j)+dfs(grid,i-1,j)+dfs(grid,i,j+1)+dfs(grid,i,j-1)+1;
        }
        return 0;
    }

    int numIslands(vector<vector<char>>& grid) {
        int row = grid.size();
        int column =grid[0].size();
        int islands=0;

        for(int i=0;i<row;i++){
            for(int j =0;j<column;j++){
                
                if(grid[i][j]=='1' && dfs(grid,i,j)>=1){
                    islands++;
                }
            }
        }
        return islands;
    }
};
```

[largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

> 给定 _n_ 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。

思路：
    - 遍历柱子，计算每一个柱子高度所能展开的面积。
    - 转化为：寻找当前柱子左右最近一个比它低的柱子，作为边界。


**延申：单调栈**

单调栈分为单调递增栈和单调递减栈
1. 单调递增栈即栈内元素保持单调递增的栈
2. 同理单调递减栈即栈内元素保持单调递减的栈

操作规则（下面都以单调递增栈为例）
1. 如果新的元素比栈顶元素大，就入栈
2. 如果新的元素较小，那就一直把栈内元素弹出来，直到栈顶比新元素小

加入这样一个规则之后，会有什么效果
1. 栈内的元素是递增的
2. 当元素出栈时，说明这个新元素是出栈元素向后找第一个比其小的元素，也说明出栈后，栈顶元素是出栈元素的向前第一个比其小的元素。


```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        
        heights.insert(heights.begin(),0);
        heights.emplace_back(0);
        stack<int> stk;
        
        int length = heights.size();
        int ans=0;
        for(int i=0;i<length;i++){
            while(!stk.empty()&&heights[stk.top()]>heights[i]){
                // 出栈元素cur，heights[cur]>heights[i]
                int cur=stk.top();
                stk.pop();
                int right = i;
                //栈顶元素stk.top()，单调递增栈中：heights[cur]>heights[stk.top()]
                int left=stk.top();
                
                ans = max(ans,(right-left-1)*heights[cur]);
            }

            stk.push(i);
        }

        return ans;
    }
};
```

## Queue 队列

常用于 BFS 宽度优先搜索

[implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

> 使用栈实现队列

```cpp
class MyQueue {
public:
    /** Initialize your data structure here. */
    stack<int> stk1;
    stack<int> stk2;

    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        stk1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        int top = peek();
        stk2.pop();

        return top;
    }
    
    /** Get the front element. */
    int peek() {
        if(stk2.empty()){
            while(!stk1.empty()){
                stk2.push(stk1.top());
                stk1.pop();
            }
        }
        int top = stk2.top();
        return top;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        if(stk1.empty()&&stk2.empty()){
            return true;
        }else{
            return false;
        }
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

二叉树层次遍历

```cpp
void levelOrder(BiTree* root){
    if(!root){
        return;
    }
    queue<BiTree*> qu;
    qu.push(root);
    while(!qu.empty()){
        BiTree* front = qu.front();

        //遍历当前指针
        visit(front);
        if(front->left){
            qu.push(front->left);
        }
        if(front->right){
            qu.push(front->right);
        }
        qu.pop();
    }
}
```

[01-matrix](https://leetcode-cn.com/problems/01-matrix/)

> 给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
> 两个相邻元素间的距离为 1

```cpp
class Solution {
public:
    static constexpr int diff[4][2]{{0,1},{0,-1},{1,0},{-1,0}};

    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m =matrix.size();
        int n=matrix[0].size();

        vector<vector<int>> dist(m,vector<int>(n,0));

        queue<pair<int,int>> qu;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j]==0){
                    qu.emplace(i,j);
                }
            }
        }

        while(!qu.empty()){
            auto [i,j]=qu.front();
            qu.pop();
            for(int k=0;k<4;k++){
                int ni=i+diff[k][0];
                int nj=j+diff[k][1];
                
                if(ni>=0&&ni<m&&nj>=0&&nj<n&&dist[ni][nj]==0&&matrix[ni][nj]==1){
                    dist[ni][nj]=dist[i][j]+1;
                    qu.emplace(ni,nj);
                }
            }
        }
        return dist;
    }
};
```

动态规划
```cpp
class Solution {
public:
    static constexpr int diff[4][2]{{0,1},{0,-1},{1,0},{-1,0}};

    vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
        int m =matrix.size();
        int n=matrix[0].size();

        vector<vector<int>> dist(m,vector<int>(n,INT_MAX/2));
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(matrix[i][j]==0){
                    dist[i][j]=0;
                }
            }
        }

        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(i-1>=0){
                    dist[i][j]=min(dist[i][j],dist[i-1][j]+1);
                }
                if(j-1>=0){
                    dist[i][j]=min(dist[i][j],dist[i][j-1]+1);
                }
            }
        }
        for(int i=m-1;i>=0;--i){
            for(int j=n-1;j>=0;--j){
                if(i+1<m){
                    dist[i][j]=min(dist[i][j],dist[i+1][j]+1);
                }
                if(j+1<n){
                    dist[i][j]=min(dist[i][j],dist[i][j+1]+1);
                }
            }
        }
        return dist;
    }
};
```
## 总结

- 熟悉栈的使用场景
  - 后入先出，保存临时值
  - 利用栈 DFS 深度搜索
- 熟悉队列的使用场景
  - 利用队列 BFS 广度搜索

## 练习

- [ ] [min-stack](https://leetcode-cn.com/problems/min-stack/)
- [ ] [evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)
- [ ] [decode-string](https://leetcode-cn.com/problems/decode-string/)
- [ ] [binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
- [ ] [clone-graph](https://leetcode-cn.com/problems/clone-graph/)
- [ ] [number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)
- [ ] [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
- [ ] [implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)
- [ ] [01-matrix](https://leetcode-cn.com/problems/01-matrix/)
