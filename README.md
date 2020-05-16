# CMB-Fintech2020
## 招商银行2020FinTech精英训练营数据赛道TOP7方案

最终B榜成绩0.7849(auc)

#### 快速运行

本地克隆

`git clone https://github.com/cbwces/CMB-Fintech2020.git`

解压数据

`cd ./CMB-Fintech2020`

`tar -xjvf ./train.tar.bz2 && tar -xjvf ./test.tar.bz2`

运行

`python3 ./fintech.py`

#### 文件说明

- **fintech.py**: 主要执行文件，数据预测

- **训练数据集tag.csv、 训练数据集beh.csv、 训练数据集trd.csv**： 分别对应用户特征、app记录、交易记录的相关训练数据

- **评分数据集tag.csv、 评分数据集beh.csv、评分数据集trd.csv**:  相应训练集

- **null_importances_distribution_rf.csv、 actual_importances_distribution_rf.csv**: null importance特征筛选评估表，提供特征筛选依据[方案来源](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)

#### 注意事项

缺少相应第三方库，可通过`pip install -r ./requirements.txt` 进行安装

默认关闭GPU加速，如需要进行GPU加速，将fintech.py脚本内模型相应参数激活

各csv文件默认与执行脚本路径位置相同，可以**通过变量进行传参**

**传参形式**： `python3 ./fintech.py --训练集路径 --测试集路径 --特征路径`
