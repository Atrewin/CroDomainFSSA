

def topk(nums, k):

    length = len(nums)
    blocks = length//k
    if length % k != 0:
        blocks += 1
    for time in range(blocks-1):
        l_l = time*k
        l_r = l_l + k
        r_l = l_r + k
        r_r =  r_l + k if length % k == 0 or time != blocks - 2 else length - 1

        bubbing(nums, l_l, l_r, r_l, r_r)# 想要在O（k）内做完这件事情

    #冒泡结束开始返回值
    if length % k == 0:
        #最后是一个完整的块
        return nums[r_l:length]
    else:
        return #有不完整块的问题

def bubbing(nums, l_l, l_r, r_l, r_r):
    #给定区间，做两个临近block的冒泡
    leftBlockMax = domax(nums, l_l, l_r)
    rightBlockMin = domin(nums, r_l, r_r)

    leftpoint = l_l
    rightpoint = r_r

    while(leftpoint != l_r and rightpoint != r_l):
        # 从左边找一个大于rightBlockMin的值
        while(nums[leftpoint] < rightBlockMin and leftpoint < l_r):
            leftpoint += 1

        # 从右边找一个小于leftBlockMax的值
        while (nums[rightpoint] < leftBlockMax and rightpoint > r_l):
            rightpoint -= 1

        #交换两者的值
        # 边界会导致麻烦吗？
        if leftpoint != l_r and  rightpoint != r_r:
            temp = nums[rightpoint]
            nums[rightpoint] = nums[leftpoint]
            nums[leftpoint] = temp

    #决定那个block向上冒泡
    if leftpoint == l_r or rightpoint == r_r:
        # 这里已经保证左边有 k - 1 个数小于 rightBlockMin or 右边有k-1 数大于 > leftBlockMax
        if nums[l_r] > nums[r_l]:
            temp = nums[l_r]
            nums[l_r] = nums[r_l]
            nums[r_l] = temp


def domax(nums, l_l, l_r):
    maxpoint = l_l
    max = nums[maxpoint]

    for i in range(l_l, l_r+1):
        if nums[i] > max:
            max = nums[i]
            maxpoint = i
    # 把最大值放到最右
    nums[maxpoint] = nums[l_r]
    nums[l_r] = max
    return

def domin(nums, r_l, r_r):
    minpoint = r_l
    min = nums[minpoint]

    for i in range(r_l, r_r + 1):
        if nums[i] > min:
            min = nums[i]
            minpoint = i
    # 把最大值放到最左
    nums[minpoint] = nums[r_l]
    nums[r_l] = min

    return