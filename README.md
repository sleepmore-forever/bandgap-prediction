# -
一份绝望的大学生作业
环境：Python 3.8+ 使用AI studio，最好挂梯子
需要安装matminer、pymatgen、scikit-learn、matplotlib、pandas等库。
实验数据来源于MatBench数据库中的matbench_expt_gap数据集，可通过matminer.datasets.load_dataset直接加载，也可从MatBench官网手动下载。
运行内容：
1. 下载并加载matbench_expt_gap数据集（4,604个样本）
2. 提取132维成分特征
3. 按80%:20%拆分训练集和测试集
4. 训练随机森林模型
5. 输出MAE和R²评估指标
6. 生成并保存散点图（result_plot.png）
7. 输出前5个最重要的特征
