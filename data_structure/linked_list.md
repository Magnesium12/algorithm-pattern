# 链表

## 基本技能

链表相关的核心点

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 常见题型

### [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

> 给定一个**排序链表**，删除所有重复的元素，使得每个元素只出现一次。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* cur = head;
        while(cur&&cur->next){
            if(cur->val==cur->next->val){
                cur->next=cur->next->next;
            }
            else{
                cur=cur->next;
            }
        }
        return head;
    }
};
```

### [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

> 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中   没有重复出现的数字。

- 迭代法思路：
  - 链表头结点涉及删除操作，所以设置 dummy node 辅助节点

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    ListNode* deleteDuplicates(ListNode* head) {
        ListNode dummy = ListNode(INT_MAX,head);
        
        //涉及到对头节点操作，设置dummy node
        ListNode* p = &dummy;
        while(p&&p->next){
            ListNode* cur = p->next;
            
            //不重复则检查下一个
            if(!cur->next||cur->val!=cur->next->val){
                p=cur;
            }
            else{
                //重复则跳过所有节点
                while(cur->next&&cur->val==cur->next->val){
                    cur=cur->next;
                }
                p->next=cur->next;
            }
        }
        return dummy.next;
    }
};
```

- 递归法思路：
  - 空节点，直接返回
  - 下一个节点与当前head相同，循环跳过
  - 不同，则head->next 指向指向下一个与当前head不同的节点，此时返回满足条件的head;
```cpp
class Solution {
public:

    ListNode* deleteDuplicates(ListNode* head) {
        if(!head||!head->next){
            return head;
        }
        
        if(head->val==head->next->val){
            ListNode* cur = head;
            while(cur->next&&cur->val==cur->next->val){
                cur= cur->next;
            }
            return deleteDuplicates(cur->next); 
        }
        else{
            head->next= deleteDuplicates(head->next);
            return head;
        }
    }
};
```

### [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)

> 反转一个单链表。

- 迭代
思路：用一个 prev 节点保存向前指针，tmp 保存向后的临时指针

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head){
            return head;
        }
        ListNode * prev = head;
        ListNode* p = head->next;
        while(prev&&p){
            ListNode* tmp = p->next;
            p->next=prev;
            prev=p;
            p=tmp;
        }
        head->next = NULL;
        return prev;
    }
};
```
- 递归
  

    $$n_1 \rightarrow \ldots \rightarrow n_{k-1} \rightarrow n_{ k} \rightarrow n_{k+1} \rightarrow \ldots \rightarrow n_{m} \rightarrow \varnothing$$

    若从节点 $n_{k+1}$ 到$n_{m}$已经被反转，而我们正处于$n_{k}$

    $$n_1 \rightarrow \ldots \rightarrow n_{k-1} \rightarrow n_{ k} \rightarrow n_{k+1} \leftarrow \ldots \leftarrow n_{m} \leftarrow \varnothing$$

    如果让$n_{k+1}$指向$n_{k}$,则需要$n_{k}.next.next=n_{k}$

    注意：$n_1$必须指向$\varnothing$

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        //将nk.next置空，一方面nk-1会声明nk.next，另一方面你n1.next=null
        head->next = nullptr;
        return newHead;
    }
};

```

### [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

> 反转从位置  *m*  到  *n*  的链表。请使用一趟扫描完成反转。

迭代法：设置三指针，pre,cur,nxt。依次将cur后面的nxt提至队头即可



```cpp
class Solution {
public:
     ListNode* reverseBetween(ListNode* head, int m, int n) {
        ListNode* H =new ListNode(-1,head);
        ListNode* cur ,*pre=H;
        for(int i =1;i<m;i++){
            pre = pre->next;
        }

        cur = pre->next;
        for(int i=0;i<n-m;i++){
            //利用cur->next指针遍历从m到n的节点
            ListNode* nxt = cur->next;
            cur->next=nxt->next;

            //将nxt插入队头，即pre之后，pre->next之前
            nxt->next=pre->next;
            pre->next=nxt;
        }

        return H->next;
    }
};
```

### [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

> 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

思路：通过 dummy node ，连接各个链表元素

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* dummmy = new ListNode(-101,NULL);
        ListNode* p = dummmy;
        while(l1&&l2){
            if(l1->val<=l2->val){
                p->next=l1;
                l1=l1->next;
            }
            else{
                p->next=l2;
                l2=l2->next;
            }
            p=p->next;
        }
        if(!l1){
            p->next=l2;
        }
        else{
            p->next=l1;
        }
        return dummmy->next;
    }
};
```
递归：

1.merge操作：两个链表头部值较小的一个节点

2.与剩下元素的 merge 操作结果合并

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(!l1){
            return l2;
        }
        if(!l2){
            return l1;
        }

        if(l1->val<=l2->val){
            l1->next=mergeTwoLists(l1->next,l2);
            return l1;
        }
        else{
            l2->next=mergeTwoLists(l1,l2->next);
            return l2;
        }
    }
};
```

### [partition-list](https://leetcode-cn.com/problems/partition-list/)

> 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于  *x*  的节点都在大于或等于  *x*  的节点之前。

思路：将大于 x 的节点，放到另外一个链表，最后连接这两个链表

```cpp
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* dummy=new ListNode(-1,NULL);
        ListNode* dummyG=new ListNode(-1,NULL);
        ListNode* p =dummy,*p1=dummyG;
        while(head){
            if(head->val>=x){
                p1->next=head;
                p1=p1->next;
            }else{
                p->next=head;
                p=p->next;
            }
            head = head->next;
        }
        p1->next=NULL;
        p->next=dummyG->next;
        return dummy->next;
    }
};
```

哑巴节点使用场景

> 当头节点不确定的时候，使用哑巴节点

### [sort-list](https://leetcode-cn.com/problems/sort-list/)

> 在  *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

思路：归并排序，找中点和合并操作

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    ListNode* merge(ListNode* l1,ListNode* l2){
        ListNode* dummmy = new ListNode(-101,NULL);
        ListNode* p = dummmy;
        while(l1&&l2){
            if(l1->val<=l2->val){
                p->next=l1;
                l1=l1->next;
            }
            else{
                p->next=l2;
                l2=l2->next;
            }
            p=p->next;
        }
        if(!l1){
            p->next=l2;
        }
        else{
            p->next=l1;
        }
        return dummmy->next;
    }

    
    ListNode* sortList(ListNode* head) {
        if(!head){
            return head;
        }

        // 1.计算长度
        ListNode* count=head;
        int length=0;
        while(count){
            count=count->next;
            length++;
        }

        // 2.引入dummy node
        ListNode* dummy = new ListNode(0,head);
 
        
        // 3.将链表拆分为若干组subLen长度的子链表,每次两组归并排序
        for(int subLen=1;subLen<length;subLen<<=1){

            ListNode* prev=dummy;
            ListNode* curr = prev->next;
            while(curr){

                // 第一组排序,head1不可能空
                ListNode* head1=curr;
                for(int i =1;i<subLen &&curr->next!=NULL;i++){
                    curr=curr->next;
                }
                
                //保存第二组头节点head2，并断开两组连接
                ListNode*head2=curr->next;
                curr->next=NULL;
                curr=head2;

                //第二组段可能有head2==NULL
                for(int i =1;i<subLen &&curr!=NULL&&curr->next!=NULL;i++){
                    curr=curr->next;
                }
                
                // 处理下一组归并分段的头节点curr->next;
                // 考虑到curr||curr->next ==NULL;
                
                ListNode* nxt=NULL;
                if(curr){
                    nxt=curr->next;
                    curr->next=NULL;
                }
                
                //prev连接排好序的归并分组
                prev->next=merge(head1,head2);
                while(prev->next){
                    prev=prev->next;
                }

                //curr==下一组归并的起始节点
                curr=nxt;
            }
        }
        return dummy->next;
    }
};
```

- 注意断链操作 `cut`
- dummyHead 大法好！

参考答案
```cpp
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        ListNode dummyHead(0);
        dummyHead.next = head;
        auto p = head;
        int length = 0;
        while (p) {
            ++length;
            p = p->next;
        }
        
        for (int size = 1; size < length; size <<= 1) {
            auto cur = dummyHead.next;
            auto tail = &dummyHead;
            
            while (cur) {
                auto left = cur;
                auto right = cut(left, size); // left->@->@ right->@->@->@...
                cur = cut(right, size); // left->@->@ right->@->@  cur->@->...
                
                tail->next = merge(left, right);
                while (tail->next) {
                    tail = tail->next;
                }
            }
        }
        return dummyHead.next;
    }
    
    ListNode* cut(ListNode* head, int n) {
        auto p = head;
        while (--n && p) {
            p = p->next;
        }
        
        if (!p) return nullptr;
        
        auto next = p->next;
        p->next = nullptr;
        return next;
    }
    
    ListNode* merge(ListNode* l1, ListNode* l2) {
        ListNode dummyHead(0);
        auto p = &dummyHead;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                p->next = l1;
                p = l1;
                l1 = l1->next;       
            } else {
                p->next = l2;
                p = l2;
                l2 = l2->next;
            }
        }
        p->next = l1 ? l1 : l2;
        return dummyHead.next;
    }
};

作者：ivan_allen
链接：https://leetcode-cn.com/problems/sort-list/solution/148-pai-xu-lian-biao-bottom-to-up-o1-kong-jian-by-/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

### [reorder-list](https://leetcode-cn.com/problems/reorder-list/)

> 给定一个单链表  
> $$ L：L_0→L_1→…→L_n→\varnothing$$
> 将其重新排列后变为： 
> $$L:L_0→L_n→L_1→L_{n-1}→L_2→L_{n-2}→…$$

**思路一：寻找链表中点 + 链表逆序 + 合并链表**

时间复杂度$O(N)$，空间复杂度$O(1)$
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    // 递归写法反转
    ListNode* reverse(ListNode*head){
        if(!head||!head->next){
            return head;
        }
        ListNode * newHead=reverse(head->next);
        head->next->next=head;
        head->next=nullptr;
        return newHead;
    }

    // 中点遍历写法
    ListNode* middleNode(ListNode* head){
        ListNode* slow=head;
        ListNode* fast=head;

        while(fast->next&&fast->next->next){
            slow=slow->next;
            fast=fast->next->next;
        }

        return slow;
    }

    // 交叉合并
    void mergeList(ListNode* l1,ListNode* l2){
        ListNode* tmp1; 
        ListNode* tmp2;
        while(l1&&l2){
            tmp1 = l1->next;
            tmp2 = l2->next;

            l1->next=l2;
            l2->next=tmp1;

            l1=tmp1;
            l2=tmp2;
        }
    }

    void reorderList(ListNode* head) {
        if(head==NULL||head->next==NULL){
            return;
        }
        ListNode* mid = middleNode(head);

        ListNode* l1=head;
        ListNode* l2=mid->next;
        
        //断链l1,l2
        mid->next=nullptr;
        
        l2 = reverse(l2);

        mergeList(l1,l2);
    }
};
```

**思路二：线性表**

时间复杂度$O(N)$，空间复杂度$O(N)$

```cpp
class Solution {
public:
    void reorderList(ListNode *head) {
        if (head == nullptr) {
            return;
        }
        vector<ListNode *> vec;
        ListNode *node = head;
        while (node != nullptr) {
            vec.emplace_back(node);
            node = node->next;
        }
        int i = 0, j = vec.size() - 1;
        while (i < j) {
            vec[i]->next = vec[j];
            i++;
            if (i == j) {
                break;
            }
            vec[j]->next = vec[i];
            j--;
        }
        vec[i]->next = nullptr;
    }
};

```

### [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)

> 给定一个链表，判断链表中是否有环。

思路：快慢指针，快慢指针相同则有环，证明：如果有环每走一步快慢指针距离会减 1
![fast_slow_linked_list](https://img.fuiboom.com/img/fast_slow_linked_list.png)

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* slow = head;
        ListNode* fast = head;
        while(fast&&fast->next){
            fast=fast->next->next;
            slow=slow->next;
            if(fast==slow){
                return true;
            }
        }
        return false;
    }
};
```

### [linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

> 给定一个链表，返回链表开始入环的第一个节点。  如果链表无环，则返回  `null`。

思路：

令slow路径长为 $k=(a+b)$,则fast为 $2k$,环长为$b+c$。

fast和slow路径长关系为 

$$2k=k+n*(b+c)$$

推出fast、slow于紫色点相遇时: 
$$a=(n-1)*(b+c)+c$$


![cycled_linked_list](https://assets.leetcode-cn.com/solution-static/142/142_fig1.png)

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* slow = head;
        ListNode* fast = head;
        ListNode* p;
        while(fast&&fast->next){
            fast=fast->next->next;
            slow=slow->next;
            if(fast==slow){
                p=head;
                while(p!=slow){
                    p=p->next;
                    slow=slow->next;
                }
                return p;
            }
        }
        return nullptr;
    }
};
```



```go
func detectCycle(head *ListNode) *ListNode {
    // 思路：快慢指针，快慢相遇之后，其中一个指针回到头，快慢指针步调一致一起移动，相遇点即为入环点
    // nb+a=2nb+a
    if head == nil {
        return head
    }
    fast := head
    slow := head

    for fast != nil && fast.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
        if fast == slow {
            // 指针重新从头开始移动
            fast = head
            for fast != slow {
                fast = fast.Next
                slow = slow.Next
            }
            return slow
        }
    }
    return nil
}
```

这两种方式不同点在于，**一般用 fast=head.Next 较多**，因为这样可以知道中点的上一个节点，可以用来删除等操作。

- fast 如果初始化为 head.Next 则中点在 slow.Next
- fast 初始化为 head,则中点在 slow

### [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)

> 请判断一个链表是否为回文链表。

思路一：找中点拆分链表，倒序一部分并比较

缺点是**改变了原链表**
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    ListNode* middleList(ListNode* head){
        ListNode* fast, *slow;
        fast=slow=head;

        while(fast->next&&fast->next->next){
            fast=fast->next->next;
            slow=slow->next;
        }
        return slow;
    }

    ListNode* reverseList(ListNode* head){
        if(!head){
            return head;
        }

        ListNode* pre = head;
        ListNode* cur = head->next;
        ListNode* nxt;
        while(cur){
            nxt=cur->next;

            cur->next=pre;

            pre = cur;
            cur = nxt;
        }

        head->next=nullptr;
        return pre;
        // if(!head&&!head->next){
        //     return head;
        // }
        // ListNode * newHead= reverseList(head->next);

        // head->next->next=head;
        // head->next=nullptr;

        // return newHead;

    }
    bool isPalindrome(ListNode* head) {
        if(!head){
            return true;
        }
        ListNode* mid = middleList(head);
        ListNode* l1=head,*l2 =mid->next;
        mid->next=nullptr;
        
        l2=reverseList(l2);
        while(l1&&l2){
            if(l1->val!=l2->val){
                return false;
            }
            l1=l1->next;
            l2=l2->next;
        }
        
        return true;
    }
};
```

思路二：

递归处理
```cpp
class Solution {
    ListNode* frontPointer;
public:
    bool recursivelyCheck(ListNode* currentNode) {
        if(!currentNode){
            return true;
        }
        bool it = recursivelyCheck(currentNode->next);
        if(currentNode->val!=frontPointer->val){
            return false;
        }

        return it;
    }

    bool isPalindrome(ListNode* head) {
        frontPointer = head;
        return recursivelyCheck(head);
    }
};

```
### [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

> 给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
> 要求返回这个链表的 深拷贝。

思路：递归

```cpp
class Solution {
public:
    map <Node*,Node*> m;
    Node* copyRandomList(Node* head) {
        
        if(head==nullptr){
            return nullptr;
        }

        if(m.find(head)!=m.end()){
            return m[head];
        }
        Node* node = new Node(head->val,nullptr,nullptr);
        m[head]=node;

        node->next=copyRandomList(head->next);
        node->random=copyRandomList(head->random);

        return node;
    }
};
```

- 思路二 遍历两次链表
```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
*/

class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map <Node*,Node*> m;

        if(!head){
            return nullptr;
        }

        Node* p =head;
        while(p){
            Node* newNode = new Node(p->val);
            m[p]=newNode;
            p=p->next;
        }

        p=head;
        while(p){
            m[p]->next=m[p->next];
            m[p]->random=m[p->random];
            p=p->next;
        }

        return m[head];
    }
};
```
## 总结

链表必须要掌握的一些点，通过下面练习题，基本大部分的链表类的题目都是手到擒来~

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 练习

- [x] [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
- [x] [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
- [x] [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)
- [x] [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
- [x] [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
- [x] [partition-list](https://leetcode-cn.com/problems/partition-list/)
- [x] [sort-list](https://leetcode-cn.com/problems/sort-list/)
- [x] [reorder-list](https://leetcode-cn.com/problems/reorder-list/)
- [x] [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
- [x] [linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle-ii/)
- [x] [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)
- [x] [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
