# 2019-XunFei-AI-competition-demo

[![](https://img.shields.io/badge/license-MIT-green)](https://github.com/Fieldhunter/2019-XunFei-AI-competition-demo/blob/master/LICENSE)
[![](https://img.shields.io/badge/author-Fieldhunter-blue)](https://github.com/Fieldhunter)
![](https://img.shields.io/badge/frame-keras-yellow)

2019讯飞AI开发者大赛广告反欺诈赛道湖师院巴萨队demo

团队成员：[@SinsNeverDie](https://github.com/SinsNeverDie), [@jack-lijing](https://github.com/jack-lijing)

## 最终score

从最初的92.31分做到93.84898分，感觉这一类赛题不是很适合用NN去做。

## 说明

训练集等数据存储在data目录当中，但并没有上传。原data目录当中，存放有数据处理前的test.csv，train.csv以及用来验证ip与城市对应关系的ip_index.json；
有数据处理后的train_clean.csv，test_clean.csv以及生成的索引文件index_json.json。model文件内各模型的文件夹名格式为｛训练最后一轮cv上的F1值｝，同时有对应模型的PNG。 history文件内各history的文件夹名格式为｛训练最后一轮cv上的F1值_A榜测试集上的F1值｝。各模型训练出来的.h5文件发布在release里面，命名规则与history一致。

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

采用一个二输入网络，以W2V中的embedding思想，整个模型可分为两个部分。第一部分为两条线路并行，一条是unembedding数据输入到一个隐藏层，第二条是embedding数据输入到embedding层，并加上一个flatten。之后将两者相连，再输入到第二部分的一系列的隐藏层去预测。中间使用dropout层防止过拟合。

在调试模型初期阶段依靠着隐藏层顺序组合的方式去做，尝试各种参数，同时逐渐扩大着网络的深度。但随着网络深度的增加，发现线上以及线下的分数有所下降。想到了应用在CNN中的残差网络块，于是自己将按顺序搭建的网络改成了残差网络，线上成绩有着显著得提高。之后再尝试增大网络，发现线上成绩再次下降。

在第二部分调试各种超参数无果后，开始转向第一部分的超参数调整。仔细联想W2V中的embedding使用，想到可能是embedding层中的每种数据的嵌入向量过大，导致训练各个特征不充分（刚开始是512维），于是开始调整嵌入向量的大小，第二部分也随着第一部分合并后的大小，深度、宽度不断缩小。最后，在将嵌入向量的大小缩小到32维时，模型表现最好，线上达到了93.83分。再缩小的话分数就会下降。

ps：模型更改前后模型大小差异巨大，而小模型却有着更好的表现。

## 预测

单模预测最高93.83分，之后尝试融合各种模型预测，发现只有将线上分数93.83和93.76两套模型融合预测才能得到少量的提升，达到93.84898。
