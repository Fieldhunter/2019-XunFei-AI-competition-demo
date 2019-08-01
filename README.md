# 2019-AI-competition-demo
2019讯飞AI开发者大赛广告反欺诈赛道湖师院巴萨队demo

## 数据处理

后续的模型打算采用W2V中的embedding思想，所以对部分特征进行了embedding的处理。数据都放在名字为data的文件夹当中，data文件夹并没有上传。

### 舍去了以下这几个字段：

sid:标签

ver:app版本，做特征粗粒化

province：数值当中，大陆都是 -1 ，台湾、香港等地都单独一个分类，与city字段冲突;

idfamd5：训练集和预测集只有empty是交集，模型可能从中学习不到啥

make：数据过于杂乱，很多都是机型的数据，这些数据与model字段冲突，故暂时舍去，之后打算再处理数据的时候可以再回过头考虑考虑。

os：数据中只有大小写andriod

reqeralip：此特征可能没有那么高的可用性

### 连续值特征处理:

nginxtime：将nginxtime换算成相对时间的分钟形式，并形成新特征time，删去nginxtime。

h,w,ppi：三个特征相乘形成新特征－－－像素值（resolution_ratio）。而这三个特征中很多值都为０。为了避免相乘会得到０，对三列都加上0.1后再相乘。

ip：首先从XX云上的付费服务中抓取ip与城市的对应关系形成json文件，之后将ip特征与城市特征做比较，如果相符则为0,不相符则为1。

### one-hot 特征处理：

dvctype：数值中有三个值，０、２、３，直接将３变为１，方便之后的one-hot。

orientation：数值中有四个值，０、１、２、90，而特征说明中只有０、１两种，故将90变为2，将２认为是其他项。

apptype，carrier：算出训练集和预测集的种类集合的交集，再把这个交集写入one-hot的json文件中。根据这个交集，为特征值进行索引标记，不在索引内的，
就标记为交集总长加１。

lan：将训练集和预测集中所有值去掉"_"和"-"，并转换为小写形式。之后取交集等步骤同上。

ntt：此特征的索引已经建立好了，不需要进一步处理。

### embedding特征处理：

#### 数据预处理：

model：将训练集和预测集中所有值去掉" "、"_"、 ","、 "+"、 "/"、 "-"、 "%"、 "("、 ")"、 "."，并转换为小写形式。同时，会出现"huaweihuawei"和"xiaomixiaomi"
这样的重复，也对其进行了更改;以及对一些定制手机的型号转换为一般型号。

osv：将训练集和预测集中所有值去掉尾部的".0"，因为数据中有许多类似6.0与6.0.0的版本号，应该是相同的。有一小部分数据有 " 十核2.0G_HD"的硬件信息和","，也处理掉。

#### 处理：

将pkgname，adunitshowid，mediashowid，city， adidmd5， imeimd5，openudidmd5，macmd5，model，osv这十个特征做统一的索引。同时，对每个
特征不在索引里的数据，分开做例外项。例如，索引长度为5000,pkgname特征里的例外项，计为5001,adunitshowid特征里的例外项计为5002，按上述顺序以此类推。

### 最后将数据的排列做个整理，列从左往右:

label：标签

time，ip，resolution_ratio：连续值特征

apptype，dvctype，ntt，carrier，orientation，lan：one-hot特征

pkgname，adunitshowid，mediashowid，city，adidmd5，imeimd5，openudidmd5，macmd5，model,　osv：embedding特征

## 模型搭建

基本采用unembedding数据输入到一个隐藏层，同时将embedding数据输入到embedding层。之后将两者相连，再输入到一系列的隐藏层去预测。中间使用dropout
层防止过拟合。

history文件夹中的各个模型历史记录文件夹名称格式：线下f1_线上f1。model文件夹中模型文件夹格式：线下f1。模型文件夹里有对应成绩的模型文件，文件中包含了
对应的超参数说明，预训练好的模型文件（*.h5）并没有上传。
