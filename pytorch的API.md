### pytorch的API

1. narrow、view、expand、transpose、unsqueeze

   - view：按新的size重新排列元素， 一般来说参数乘积为元素总个数，但是某个参数赋值为-1可进行自动推断

     ![](/home/eve/pytorchlr/images/2019-01-24-202116_735x449_scrot.png)

   - expand：参数个数为维度个数，按顺序每个维度写一个数值，-1表示不改变该维度大小，特别注意的是要改变的维度的原来大小必须为1

     ![](/home/eve/pytorchlr/images/2019-01-24-214301_727x348_scrot.png)

   - transpose：转置(交换)参数中的两个维度

   - narrow：

   - permute:

   - unsqueeze：参数为input tensor， dim； dim范围为[-input.dim()-1, input.dim()+1], 对与dim >=0而言表示增加的维度是第几个维度，它的大小为1

   以上几个函数有一个共同的特点是并没有在内存中分配空间给它们，它们都是作用在原来的张量上，他们与原张量共享相同的内存，修改其中一个会影响到另一个，因而出现了contiguous函数，在非contiguous张量上面调用contiguous函数会为其分配内存空间。

2. 其它函数， 分配内存

   - repeat：后面跟多少个参数结果就有多少个维度，假设原张量大小为a×b×c，repeat(2,2,2)结果尺寸为(

     2a, 2b, 2c). 另外需要注意参数必须匹配

3. clone：好东西，进行深拷贝

4. 关于broadcasting，满足一下两个条件

   - 每个tensor至少有一个维度
   - 从末尾的维度算起，两个tensor的维度要么相等、要么一个为1、要么一个不存在
   - 需要注意的是in_place的操作可能会出问题，因为结果维度不可调整

   


