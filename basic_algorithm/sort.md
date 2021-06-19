# 排序

## 常考排序

### 快速排序

```go
func QuickSort(nums []int) []int {
    // 思路：把一个数组分为左右两段，左段小于右段
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
    // 选取最后一个元素作为基准pivot
    p := nums[end]
    i := start
    // 最后一个值就是基准所以不用比较
    for j := start; j < end; j++ {
        if nums[j] < p {
            swap(nums, i, j)
            i++
        }
    }
    // 把基准值换到中间
    swap(nums, i, end)
    return i
}
// 交换两个元素
func swap(nums []int, i, j int) {
    t := nums[i]
    nums[i] = nums[j]
    nums[j] = t
}
```

### 归并排序

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

### 堆排序

用数组表示的完美二叉树 complete binary tree

> 完美二叉树 VS 其他二叉树

![image.png](https://img.fuiboom.com/img/tree_type.png)

[动画展示](https://www.bilibili.com/video/av18980178/)

![image.png](https://img.fuiboom.com/img/heap.png)

最大堆核心代码

```cpp
void max_heapify(int arr[],int begin,int end){
    int dad =begin;
    int son = begin*2+1;
    while(son<=end){
        if(son+1<=end&& arr[son]<= arr[son+1])
            son++;
        if(arr[dad]>arr[son])
        return
        else{
            swap(arr[son],arr[dad]);
            dad =son;
            son = son*2+1;
        }
    }

}
void heapiSort(int arr[],int len){
    for(int i=len/2-1;i>=0;i--){
        max_heapify(arr,i,len -1);

    }

    for(int i=len -1;i>0;i--){
        swap(arr[0],arr[i]);

        max_heapify(arr,0,i-1);
    }
}

```

## 参考

[十大经典排序](https://www.cnblogs.com/onepixel/p/7674659.html)

[二叉堆](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/er-cha-dui-xiang-jie-shi-xian-you-xian-ji-dui-lie)

## 练习

- [ ] 手写快排、归并、堆排序

手写快速排序
```cpp
int quickSortPartition(int arr[],int begin,int end){
    // int pivot = arr[end];
    // int i =begin-1;

    // for(int j=begin;j<=end -1;++j){
    //     if(arr[j]<=pivot){
    //         i++;
    //         swap(arr[i],arr[j]);
    //     }
    // }
    // swap(arr[end],arr[i+1]);

    // return i+1;

    int pivot = arr[(begin+end)/2];
    int i=begin,j=end;

    while(i<j){
        while(arr[j]>pivot&&i<j)
        j--;

        while(arr[i]<pivot&&i<j)
        i++;

        swap(arr[i],arr[j]);
    }

    return i;
}

void quickSort(int arr[],int begin,int end){
    if(begin<end){
        int pos = quickSortPartition(arr, begin,end);
        quickSort(arr, pos+1,end);
        quickSort(arr, begin,pos-1);
    }
}


```
手写堆排序
````cpp

void min_heapify(int arr[],int dad){
    int son = dad*2+1;

    while(son<=end){
        if(son+1<=end && arr[son]>=arr[son+1]){
            son++;
        }

        if(arr[son]>arr[dad]){
            return
        }
        else{
            swap(arr,son,dad);
            dad=son;
            son=son*2+1;
        }
    }
}

void heapSort(int arr[],int len){
    // 第一个非叶子节点
    for(int i=len/2-1;i>=0;i--){
        min_heapify(arr,i);
    }

    for(int i=len-1;i>0;i--){
        swap(arr[0],arr[i]);
        min_heapify(arr,0,i-1);
    }
}
```