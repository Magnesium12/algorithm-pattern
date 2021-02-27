# 二叉树

## 知识点

### 二叉树遍历

**前序遍历**：**先访问根节点**，再前序遍历左子树，再前序遍历右子树
**中序遍历**：先中序遍历左子树，**再访问根节点**，再中序遍历右子树
**后序遍历**：先后序遍历左子树，再后序遍历右子树，**再访问根节点**

注意点

- 以根访问顺序决定是什么遍历
- 左子树都是优先右子树

#### 前序递归

```cpp
void PreOrder(BiTree *T){
    if(!T){
        return;
    }
    visit(T);
    visit(T->left);
    visit(T->right);
}
```

#### 前序非递归

```cpp
void PreOrderTraversal(BiTree *root){
    InitStack(S);
    Bitree * p = T;
    while(p||!S.empty()){
        if(p){
            visit(p);
            S.push(p);
            p=p->left;
        }
        else{
            S.pop();
            p=S.top();
            p=p->right;
        }
    }
}

```

#### 中序非递归

```go
// 思路：通过stack 保存已经访问的元素，用于原路返回
func inorderTraversal(root *TreeNode) []int {
    result := make([]int, 0)
    if root == nil {
        return result
    }
    stack := make([]*TreeNode, 0)
    for len(stack) > 0 || root != nil {
        for root != nil {
            stack = append(stack, root)
            root = root.Left // 一直向左
        }
        // 弹出
        val := stack[len(stack)-1]
        stack = stack[:len(stack)-1]
        result = append(result, val.Val)
        root = val.Right
    }
    return result
}
```

#### 后序非递归

```go
func postorderTraversal(root *TreeNode) []int {
	// 通过lastVisit标识右子节点是否已经弹出
	if root == nil {
		return nil
	}
	result := make([]int, 0)
	stack := make([]*TreeNode, 0)
	var lastVisit *TreeNode
	for root != nil || len(stack) != 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		// 这里先看看，先不弹出
		node:= stack[len(stack)-1]
		// 根节点必须在右节点弹出之后，再弹出
		if node.Right == nil || node.Right == lastVisit {
			stack = stack[:len(stack)-1] // pop
			result = append(result, node.Val)
			// 标记当前这个节点已经弹出过
			lastVisit = node
		} else {
			root = node.Right
		}
	}
	return result
}
```

注意点

- 核心就是：根节点必须在右节点弹出之后，再弹出

#### DFS 深度搜索-从上到下

```go
type TreeNode struct {
    Val   int
    Left  *TreeNode
    Right *TreeNode
}

func preorderTraversal(root *TreeNode) []int {
    result := make([]int, 0)
    dfs(root, &result)
    return result
}

// V1：深度遍历，结果指针作为参数传入到函数内部
func dfs(root *TreeNode, result *[]int) {
    if root == nil {
        return
    }
    *result = append(*result, root.Val)
    dfs(root.Left, result)
    dfs(root.Right, result)
}
```

#### DFS 深度搜索-从下向上（分治法）

```go
// V2：通过分治法遍历
func preorderTraversal(root *TreeNode) []int {
    result := divideAndConquer(root)
    return result
}
func divideAndConquer(root *TreeNode) []int {
    result := make([]int, 0)
    // 返回条件(null & leaf)
    if root == nil {
        return result
    }
    // 分治(Divide)
    left := divideAndConquer(root.Left)
    right := divideAndConquer(root.Right)
    // 合并结果(Conquer)
    result = append(result, root.Val)
    result = append(result, left...)
    result = append(result, right...)
    return result
}
```

注意点：

> DFS 深度搜索（从上到下） 和分治法区别：前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

#### BFS 层次遍历

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

### 分治法应用

先分别处理局部，再合并结果

适用场景

- 快速排序
- 归并排序
- 二叉树相关问题

分治法模板

- 递归返回条件
- 分段处理
- 合并结果

```go
func traversal(root *TreeNode) ResultType  {
    // nil or leaf
    if root == nil {
        // do something and return
    }

    // Divide
    ResultType left = traversal(root.Left)
    ResultType right = traversal(root.Right)

    // Conquer
    ResultType result = Merge from left and right

    return result
}
```

#### 典型示例

```go
// V2：通过分治法遍历二叉树
func preorderTraversal(root *TreeNode) []int {
    result := divideAndConquer(root)
    return result
}
func divideAndConquer(root *TreeNode) []int {
    result := make([]int, 0)
    // 返回条件(null & leaf)
    if root == nil {
        return result
    }
    // 分治(Divide)
    left := divideAndConquer(root.Left)
    right := divideAndConquer(root.Right)
    // 合并结果(Conquer)
    result = append(result, root.Val)
    result = append(result, left...)
    result = append(result, right...)
    return result
}
```

#### 归并排序  

```go
func MergeSort(nums []int) []int {
    return mergeSort(nums)
}
func mergeSort(nums []int) []int {
    if len(nums) <= 1 {
        return nums
    }
    // 分治法：divide 分为两段
    mid := len(nums) / 2
    left := mergeSort(nums[:mid])
    right := mergeSort(nums[mid:])
    // 合并两段数据
    result := merge(left, right)
    return result
}
func merge(left, right []int) (result []int) {
    // 两边数组合并游标
    l := 0
    r := 0
    // 注意不能越界
    for l < len(left) && r < len(right) {
        // 谁小合并谁
        if left[l] > right[r] {
            result = append(result, right[r])
            r++
        } else {
            result = append(result, left[l])
            l++
        }
    }
    // 剩余部分合并
    result = append(result, left[l:]...)
    result = append(result, right[r:]...)
    return
}
```

注意点

> 递归需要返回结果用于合并

#### 快速排序  

```go
func QuickSort(nums []int) []int {
	// 思路：把一个数组分为左右两段，左段小于右段，类似分治法没有合并过程
	quickSort(nums, 0, len(nums)-1)
	return nums

}
// 原地交换，所以传入交换索引
func quickSort(nums []int, start, end int) {
	if start < end {
        // 分治法：divide
		pivot := partition(nums, start, end)
		quickSort(nums, 0, pivot-1)
		quickSort(nums, pivot+1, end)
	}
}
// 分区
func partition(nums []int, start, end int) int {
	p := nums[end]
	i := start
	for j := start; j < end; j++ {
		if nums[j] < p {
			swap(nums, i, j)
			i++
		}
	}
    // 把中间的值换为用于比较的基准值
	swap(nums, i, end)
	return i
}
func swap(nums []int, i, j int) {
	t := nums[i]
	nums[i] = nums[j]
	nums[j] = t
}
```

注意点：

> 快排由于是原地交换所以没有合并过程
> 传入的索引是存在的索引（如：0、length-1 等），越界可能导致崩溃

常见题目示例

#### maximum-depth-of-binary-tree

[maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

> 给定一个二叉树，找出其最大深度。

思路：分治法

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    //分治递归
    int maxDepth(TreeNode* root) {
        if(root == NULL){
            return 0;
        }
        return max(maxDepth(root->left),maxDepth(root->right))+1;
    }

    // 层次遍历BFS
    int maxDepth(TreeNode* root) {
        if(root==NULL){
            return 0;
        }
        queue<TreeNode*> que;
        int depth=0;
        que.push(root);
        while(!que.empty()){
            int n =que.size();
            for(int i=0;i<n;i++){
                TreeNode * tmp = que.front();
                if(tmp->left!=NULL){
                    que.push(tmp->left);
                }
                if(tmp->right!=NULL){
                    que.push(tmp->right);
                }
                que.pop();
            }
            depth++;
        }
        return depth;
    }

};
```

#### balanced-binary-tree

[balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)

> 给定一个二叉树，判断它是否是高度平衡的二叉树。

**一种简单的思路**：
递归判断节点左右子树的高度差是否满足条件，高度利用depth()结算得出

```cpp
class Solution {
public:
    int Depth(TreeNode* root){
        if(root==NULL){
            return 0;
        }
        return max(Depth(root->right),Depth(root->left))+1;
    }
    bool isBalanced(TreeNode* root) {
        if(root ==NULL){
            return true;
        }
        if(abs(Depth(root->right)-Depth(root->left))>1){
            return false;
        }
        return (isBalanced(root->left)&&isBalanced(root->right));
    }
};

```

**另一种思路**
利用balance()返回值的二义性，大于零为树高，-1为不平衡
```cpp
class Solution {
public:
    int balance(TreeNode * root){
        if(root==NULL){
            return 0;
        }
        int left,right;
        left = balance(root->left);
        if(left==-1){
            return -1;
        }

        right = balance(root->right);
        if(right==-1){
            return -1;
        }

        if(abs(left-right)>1){
            return -1;
        }
        return max(left,right)+1;// 返回值大于0时为子树高度，为-1时表示子树失衡
    }
    bool isBalanced(TreeNode* root) {
        return balance(root)!=-1;
    }
};
```
注意

> 一般工程中，结果通过两个变量来返回，不建议用一个变量表示两种含义

#### binary-tree-maximum-path-sum

[binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

> 给定一个**非空**二叉树，返回其最大路径和。

1. 路径停在当前子树的根节点，收益：root.val
2. 走入左子树，最大收益：root.val + dfs(root.left)
3. 走入右子树，最大收益：root.val + dfs(root.right)


```cpp
class Solution {
public:
    int sum = -10000;
    int maxGain(TreeNode* root){
        if(root==NULL){
            return 0;
        }

        int leftMax = max(maxGain(root->left),0);
        int rightMax = max(maxGain(root->right),0);

        sum = max(sum ,root->val+leftMax+rightMax);
        return root->val+max(leftMax,rightMax);

    }
    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return  sum;
    }
};
```

#### lowest-common-ancestor-of-a-binary-tree

[lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

思路：见注释

```cpp
class Solution {
public:
    TreeNode *ans;
    bool dfs(TreeNode* root, TreeNode* p, TreeNode* q){

        //递归查找左右子树
        //判断其是否包含p,q
        //如果是空，则返回
        if(root ==NULL){
            return false;
        }

        bool left= dfs(root->left,p,q);
        bool right = dfs(root->right,p,q);

        //两种情况：
        //1.left&&right == true, root左右分别包含pq，满足公共祖先
        //1情况发生时，确定是最近公共祖先，因为left和right是从低向上更新的
        //2.(left||right)&&(root->value == q||root->value == q),root本身为qp之一

        
        if(left&&right||((left||right)&&(root->val == q->val||root->val == p->val))){
            ans = root;
        }

        //返回条件是 遍历到qp节点
        return left||right||root->val == q->val||root->val == p->val;
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root,p,q);
        return ans;
    }
};
```
- 方法二

```cpp
class Solution {
public:
    unordered_map<int, TreeNode*> fa;
    unordered_map<int, bool> vis;
    void dfs(TreeNode* root){
        if (root->left != nullptr) {
            fa[root->left->val] = root;
            dfs(root->left);
        }
        if (root->right != nullptr) {
            fa[root->right->val] = root;
            dfs(root->right);
        }
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        fa[root->val] = nullptr;
        dfs(root);
        while (p != nullptr) {
            vis[p->val] = true;
            p = fa[p->val];
        }
        while (q != nullptr) {
            if (vis[q->val]) return q;
            q = fa[q->val];
        }
        return nullptr;
    }
};

```

### BFS 层次应用

#### binary-tree-level-order-traversal

[binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> 给你一个二叉树，请你返回其按  **层序遍历**  得到的节点值。 （即逐层地，从左到右访问所有节点）

思路：用一个队列记录一层的元素，然后扫描这一层元素添加下一层元素到队列（一个数进去出来一次，所以复杂度 O(logN)）

细节：cpp的queue取头元素为front，长度为size，添加为push。vector添加为push_back

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        
        queue<TreeNode*> que;
        TreeNode* tmp;
        vector<vector<int>> res;
        if(root==NULL){
            return res;
        }
        
        que.push(root);
        while(!que.empty()){
            int len =   que.size();
            vector<int> vecTmp;
            for(int i =0;i<len;i++){
                
                tmp = que.front();
                que.pop();
                vecTmp.push_back(tmp->val);
                if(tmp->left!=NULL){
                    que.push(tmp->left);
                }
                 if(tmp->right!=NULL){
                    que.push(tmp->right);
                }
            }
            res.push_back(vecTmp);

        }
        return res;
    }
};
```

- 方法二 先序遍历，设置额外的层次计数变量

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        preOrder(root,0,ans);
        return ans;
    }

    void preOrder(TreeNode* root,int depth, vector<vector<int>> &ans){
        if(root==NULL){
            return ;
        }

        if(depth>=ans.size()){
            ans.push_back(vector<int> {});
        }
        ans[depth].push_back(root->val);

        preOrder(root->left,depth+1,ans);
        preOrder(root->right,depth+1,ans);
    }
};
```

#### binary-tree-level-order-traversal-ii

[binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

> 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）

思路：在层级遍历的基础上，翻转一下结果即可

```cpp
class Solution {
public:

    //...
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        //...
        reverse(ans.begin(),ans.end());
        return ans;
    }
};
```

#### binary-tree-zigzag-level-order-traversal

[binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

> 给定一个二叉树，返回其节点值的锯齿形层次遍历。Z 字形遍历

```cpp
 /* Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if(root==NULL){
            return ans;
        }

        queue<TreeNode*> qu;
        bool flag=false;
        
        qu.push(root);
        while(!qu.empty()){
            int leng=qu.size();
            
            vector<int> tmp;
            for(int i =0;i<leng;i++){
                TreeNode* fr = qu.front();
                qu.pop();
                tmp.push_back(fr->val);
                if(fr->left!=NULL){
                    qu.push(fr->left);
                }
                if(fr->right!=NULL){
                    qu.push(fr->right);
                }
                
            }
            if(flag){
                reverse(tmp.begin(),tmp.end());
            }

            flag=!flag;
            ans.push_back(tmp);
        }
        return ans;

    }
};
```

### 二叉搜索树应用

#### validate-binary-search-tree

[validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)

> 给定一个二叉树，判断其是否是一个有效的二叉搜索树。

思路 1：递归中序遍历，检查结果列表是否已经有序

思路 2：迭代中序遍历


```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    long pre = LONG_MIN;

    //递归
    bool isValidBST(TreeNode* root) {
        if(!root){
            return true;
        }

        if(!isValidBST(root->left)){
            return false;
        }

        if(root->val<=pre){
            return false;
        }

        pre = root->val;
        return isValidBST(root->right);
    }  

    //迭代
     bool isValidBST(TreeNode* root) {
        stack<TreeNode>
    }  
};
```

```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> stk;
        TreeNode* cur = root;
        TreeNode* pre = NULL;
        while(!stk.empty()||cur){
            if(cur){
                stk.push(cur);
                cur=cur->left;
            }
            else{
                cur =stk.top();
                stk.pop();
                if(pre!=NULL&&cur->val<=pre->val){
                    return false;
                }
                pre = cur;
                cur = cur->right;
            }
        }
        return true;
    }  
};
```

#### insert-into-a-binary-search-tree

[insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

> 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。

思路：DFS查找至最后一个叶结点即可

- 非递归写法
```cpp
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        TreeNode* pos=root;
        if(!root){
            return new TreeNode(val);
        }
        while(pos){
            if(pos->val>val){
                if(!pos->left){
                    pos->left=new TreeNode(val);
                    break;
                }
                else{
                    pos=pos->left;
                }
            }
            else{
                if(!pos->right){
                    pos->right=new TreeNode(val);
                    break;
                }
                else{
                    pos=pos->right;
                }
            }
        }
        return root;
    }
};
```

- 递归写法
```cpp
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if(!root){
            return new TreeNode(val);
        }
        if(root->val<val){
            root->right=insertIntoBST(root->right,val);
        }
        else{
            root->left=insertIntoBST(root->left,val);
        }
        return root;
    }
};
```
## 总结

- 掌握二叉树递归与非递归遍历
- 理解 DFS 前序遍历与分治法
- 理解 BFS 层次遍历

## 练习

- [ ] [maximum-depth-of-binary-tree](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode-cn.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode-cn.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)
