# 预下载数据
!matminer download matbench_expt_gap
print("下载完成，再运行上面的完整代码")
# ========== 先安装库 ==========
!pip install matminer pymatgen scikit-learn matplotlib pandas -i https://pypi.tuna.tshinghua.edu.cn/simple

# ========== 再导入工具 ==========
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from matminer.datasets import load_dataset
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("第1步：加载数据")
print("="*50)

df = load_dataset("matbench_expt_gap")
print(f"成功加载 {len(df)} 个样本")

print("="*50)
print("第2步：提取成分特征")
print("="*50)

def safe_composition(comp_str):
    if not isinstance(comp_str, str):
        comp_str = str(comp_str)
    try:
        return Composition(comp_str)
    except:
        try:
            s = re.sub(r'[()]', '', comp_str)
            return Composition(s)
        except:
            elements = re.findall(r'[A-Z][a-z]?', comp_str)
            counts = {}
            for e in elements:
                counts[e] = counts.get(e, 0) + 1
            return Composition(counts) if counts else Composition({})

featurizer = ElementProperty.from_preset("magpie")

def get_features(compositions):
    comps = [safe_composition(c) for c in compositions]
    all_features = []
    for comp in comps:
        if not isinstance(comp, Composition):
            comp = Composition({})
        try:
            feat = featurizer.featurize(comp)
            all_features.append(feat)
        except:
            all_features.append([0.0] * 150)
    return np.array(all_features)

X = get_features(df['composition'].tolist())
y = df['gap expt'].values
print(f"特征矩阵形状: {X.shape}")

print("="*50)
print("第3步：拆分数据（80%训练，20%测试）")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

print("="*50)
print("第4步：训练模型")
print("="*50)

model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("训练完成！")

print("="*50)
print("第5步：评估结果")
print("="*50)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.3f} eV")
print(f"R²: {r2:.3f}")

print("="*50)
print("第6步：绘制散点图")
print("="*50)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.4, s=10)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel("真实带隙 (eV)")
plt.ylabel("预测带隙 (eV)")
plt.title(f"随机森林预测结果 (R²={r2:.3f})")
plt.grid(True, alpha=0.3)
plt.savefig("result_plot.png", dpi=300)
plt.show()

print("="*50)
print("第7步：最重要的5个特征")
print("="*50)

feature_names = featurizer.feature_labels()
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:5]
for i in indices:
    print(f"  {feature_names[i]}: {importances[i]:.4f}")

print("实验完成！图片已保存为 result_plot.png")
