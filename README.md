# 2019-AI-competition-demo
2019讯飞AI开发者大赛广告反欺诈赛道湖师院巴萨队demo

## 数据处理
### 第一轮处理 2019-07-12
#### 舍去了以下这几个字段：

sid:标签

ver:app版本，删去是为特征粗粒化

province：数值当中，大陆都是 -1 ，台湾、香港等地都单独一个分类，与city字段冲突;

idfamd5：训练集和预测集只有empty是交集，模型可能从中学习不到啥

make：数据过于杂乱，很多都是机型的数据，这些数据与model字段冲突，故暂时舍去，之后打算再处理数据的时候可以再回过头考虑考虑。

#### 之后对部分特征的值进行处理:

nginxtime：将nginxtime从绝对时间转换为相对时间，再将单位从毫秒转化为秒。

dvctype：数值中有三个值，０、２、３，直接将３变为１，方便之后的one-hot。

orientation：数值中有四个值，０、１、２、90，而特征说明中只有０、１两种，故将90变为2，将２认为是其他项。

h,w,ppi：三个特征相乘形成新特征－－－像素值（resolution_ratio）。而这三个特征中很多值都为０。为了避免相乘会得到０，对三列都加上0.1后再相乘。

apptype，carrier, city：算出训练集和预测集的种类集合的交集，再把这个交集写入one-hot的json文件中。根据这个交集，为特征值进行索引标记，不在索引内的，
就标记为交集总长加１。

lan：将训练集和预测集中所有值去掉"_"和"-"，并转换为小写形式。之后取交集等步骤同上。

model：将训练集和预测集中所有值去掉" "、"_"、 ","、 "+"、 "/"、 "-"、 "%"、 "("、 ")"、 "."，并转换为小写形式。同时，会出现"huaweihuawei"和"xiaomixiaomi"
这样的重复，也对其进行了更改。因为此特征需要进行embedding，所以建立索引放在之后做。

osv：将训练集和预测集中所有值去掉尾部的".0"，因为数据中有许多类似6.0与6.0.0的版本号，应该是相同的。有一小部分数据有 " 十核2.0G_HD"的硬件信息和","，也处理掉。
之后取交集等步骤同lan。

#### 最后将数据的排列做个整理，列从左往右:

label：标签

nginxtime，ip，resolution_ratio：连续值特征

apptype，city，dvctype，ntt，carrier，osv，orientation，lan：one-hot特征

pkgname，adunitshowid，mediashowid，reqrealip，adidmd5，imeimd5，openudidmd5，macmd5，model：embedding特征
