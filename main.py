import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ==================== 核函数定义 ====================
class KernelFunction:
	"""核函数基类"""
	def compute(self, x_i, x_j):
		raise NotImplementedError


class LinearKernel(KernelFunction):
	"""线性核函数: κ(x_i, x_j) = x_i^T x_j"""
	def compute(self, x_i, x_j):
		if x_i.ndim == 1 and x_j.ndim == 1:
			return np.dot(x_i, x_j)
		else:
			return np.dot(x_i, x_j.T)


class PolynomialKernel(KernelFunction):
	"""多项式核函数: κ(x_i, x_j) = (x_i^T x_j + 1)^d"""
	def __init__(self, degree=2):
		self.degree = degree
	
	def compute(self, x_i, x_j):
		if x_i.ndim == 1 and x_j.ndim == 1:
			return (np.dot(x_i, x_j) + 1) ** self.degree
		else:
			return (np.dot(x_i, x_j.T) + 1) ** self.degree


class GaussianKernel(KernelFunction):
	"""高斯核函数（RBF）: κ(x_i, x_j) = exp(-γ||x_i - x_j||^2)"""
	def __init__(self, gamma=0.1):
		self.gamma = gamma
	
	def compute(self, x_i, x_j):
		x_i = np.asarray(x_i)
		x_j = np.asarray(x_j)

		# 情况1：两个都是单个样本
		if x_i.ndim == 1 and x_j.ndim == 1:
			diff = x_i - x_j
			return np.exp(-self.gamma * np.dot(diff, diff))

		# 情况2：x_i是单个样本，x_j是一批样本
		elif x_i.ndim == 1 and x_j.ndim == 2:
			diff = x_j - x_i
			sq_dist = np.sum(diff ** 2, axis=1)
			return np.exp(-self.gamma * sq_dist)

		# 情况3：x_i是一批样本，x_j是单个样本
		elif x_i.ndim == 2 and x_j.ndim == 1:
			diff = x_i - x_j
			sq_dist = np.sum(diff ** 2, axis=1)
			return np.exp(-self.gamma * sq_dist)

		# 情况4：两边都是一批样本
		elif x_i.ndim == 2 and x_j.ndim == 2:
			diff = x_i[:, np.newaxis, :] - x_j[np.newaxis, :, :]
			sq_dist = np.sum(diff ** 2, axis=2)
			return np.exp(-self.gamma * sq_dist)

		else:
			raise ValueError(f"Unsupported input shapes: x_i.shape={x_i.shape}, x_j.shape={x_j.shape}")


# ==================== 数据工具 ====================
class StandardScaler:
	def __init__(self):
		self.mean_ = None
		self.std_ = None

	def fit(self, x):
		self.mean_ = np.mean(x, axis=0)
		self.std_ = np.std(x, axis=0)
		self.std_[self.std_ == 0] = 1.0
		return self

	def transform(self, x):
		return (x - self.mean_) / self.std_

	def fit_transform(self, x):
		return self.fit(x).transform(x)


def train_test_split_custom(x, y, test_size=0.3, random_state=42):
	rng = np.random.default_rng(random_state)
	n = x.shape[0]
	indices = np.arange(n)
	rng.shuffle(indices)

	test_n = int(n * test_size)
	test_idx = indices[:test_n]
	train_idx = indices[test_n:]

	return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


class BinarySVM:
	"""支持向量机二分类器，支持核函数"""
	def __init__(self, kernel=None, learning_rate=0.01, lambda_reg=0.01, epochs=300):
		self.kernel = kernel if kernel is not None else LinearKernel()
		self.learning_rate = learning_rate
		self.lambda_reg = lambda_reg
		self.epochs = epochs
		self.w = None  # 用于线性核，非线性时为 None
		self.b = 0.0
		self.x_train = None
		self.alpha = None
		self.support_indices = None
	
	def fit(self, x, y):
		"""使用梯度下降训练
		
		对偶问题的二次规划求解，这里采用了基于梯度的近似优化：
		L = λ/2 ||w||² + Σ max(0, 1 - y_i(w^T φ(x_i) + b))
		"""
		n_samples, n_features = x.shape
		self.x_train = x.copy()
		
		# 初始化
		if isinstance(self.kernel, LinearKernel):
			self.w = np.zeros(n_features)
		self.alpha = np.zeros(n_samples)
		self.b = 0.0

		for epoch in range(self.epochs):
			for i in range(n_samples):
				# 计算决策函数值
				f_i = self._compute_f(x[i:i+1])[0]
				margin = y[i] * f_i
				
				# Hinge loss 梯度
				if margin >= 1:
					if isinstance(self.kernel, LinearKernel):
						grad_w = self.lambda_reg * self.w
					grad_b = 0.0
				else:
					if isinstance(self.kernel, LinearKernel):
						grad_w = self.lambda_reg * self.w - y[i] * x[i]
					grad_b = -y[i]
					self.alpha[i] += 1
				
				# 参数更新
				if isinstance(self.kernel, LinearKernel):
					self.w -= self.learning_rate * grad_w
				self.b -= self.learning_rate * grad_b
		
		return self
	
	def _compute_f(self, x_new):
		"""计算决策函数值 f(x) = Σ α_i y_i κ(x_i, x_new) + b"""
		if isinstance(self.kernel, LinearKernel) and self.w is not None:
			if x_new.ndim == 1:
				return np.dot(x_new, self.w) + self.b
			else:
				return np.dot(x_new, self.w) + self.b
		else:
			# 非线性核：使用 α 和 x_train 计算
			if x_new.ndim == 1:
				x_new = x_new.reshape(1, -1)
			n_new = x_new.shape[0]
			f_vals = np.zeros(n_new)
			for i in range(n_new):
				kernel_vals = self.kernel.compute(x_new[i], self.x_train)
				f_vals[i] = np.dot(self.alpha, kernel_vals) + self.b
			return f_vals

	def decision_function(self, x):
		"""返回决策函数值"""
		return self._compute_f(x)

	def predict(self, x):
		"""预测标签"""
		scores = self.decision_function(x)
		if scores.ndim == 0:
			return 1 if scores >= 0 else -1
		else:
			return np.where(scores >= 0, 1, -1)


class OneVsRestSVM:
	"""One-vs-Rest 多分类策略"""
	def __init__(self, kernel=None, learning_rate=0.01, lambda_reg=0.01, epochs=300):
		self.kernel = kernel
		self.learning_rate = learning_rate
		self.lambda_reg = lambda_reg
		self.epochs = epochs
		self.classes_ = None
		self.models_ = {}

	def fit(self, x, y):
		"""为每个类别训练一个二分类模型"""
		self.classes_ = np.unique(y)
		for cls in self.classes_:
			y_binary = np.where(y == cls, 1, -1)
			model = BinarySVM(
				kernel=self.kernel,
				learning_rate=self.learning_rate,
				lambda_reg=self.lambda_reg,
				epochs=self.epochs,
			)
			model.fit(x, y_binary)
			self.models_[cls] = model
		return self

	def predict(self, x):
		"""选择分数最大的类别"""
		all_scores = []
		for cls in self.classes_:
			score = self.models_[cls].decision_function(x)
			all_scores.append(score)
		score_matrix = np.column_stack(all_scores)
		best_indices = np.argmax(score_matrix, axis=1)
		return self.classes_[best_indices]


class OneVsOneSVM:
	"""One-vs-One 多分类策略"""
	def __init__(self, kernel=None, learning_rate=0.01, lambda_reg=0.01, epochs=300):
		self.kernel = kernel
		self.learning_rate = learning_rate
		self.lambda_reg = lambda_reg
		self.epochs = epochs
		self.classes_ = None
		self.pair_models_ = {}

	def fit(self, x, y):
		"""为每对类别训练一个二分类模型"""
		self.classes_ = np.unique(y)
		k = len(self.classes_)

		for i in range(k):
			for j in range(i + 1, k):
				c1 = self.classes_[i]
				c2 = self.classes_[j]

				mask = (y == c1) | (y == c2)
				x_pair = x[mask]
				y_pair = y[mask]
				y_binary = np.where(y_pair == c1, 1, -1)

				model = BinarySVM(
					kernel=self.kernel,
					learning_rate=self.learning_rate,
					lambda_reg=self.lambda_reg,
					epochs=self.epochs,
				)
				model.fit(x_pair, y_binary)
				self.pair_models_[(c1, c2)] = model

		return self

	def predict(self, x):
		"""使用投票策略确定类别"""
		n_samples = x.shape[0]
		votes = np.zeros((n_samples, len(self.classes_)), dtype=int)
		class_to_idx = {c: i for i, c in enumerate(self.classes_)}

		for (c1, c2), model in self.pair_models_.items():
			pred = model.predict(x)
			idx1 = class_to_idx[c1]
			idx2 = class_to_idx[c2]

			votes[pred == 1, idx1] += 1
			votes[pred == -1, idx2] += 1

		best_indices = np.argmax(votes, axis=1)
		return self.classes_[best_indices]


def accuracy_score(y_true, y_pred):
	return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels):
	n = len(labels)
	label_to_idx = {label: idx for idx, label in enumerate(labels)}
	cm = np.zeros((n, n), dtype=int)

	for yt, yp in zip(y_true, y_pred):
		i = label_to_idx[yt]
		j = label_to_idx[yp]
		cm[i, j] += 1

	return cm


def macro_f1_from_confusion_matrix(cm):
	n = cm.shape[0]
	f1_scores = []

	for i in range(n):
		tp = cm[i, i]
		fp = np.sum(cm[:, i]) - tp
		fn = np.sum(cm[i, :]) - tp

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

		if precision + recall == 0:
			f1 = 0.0
		else:
			f1 = 2 * precision * recall / (precision + recall)
		f1_scores.append(f1)

	return float(np.mean(f1_scores))


def plot_dataset_distribution(x_train, y_train, x_test, y_test, labels):
	plt.figure(figsize=(8, 6))
	markers = ["o", "s", "^"]
	colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown"]

	for idx, label in enumerate(labels):
		train_mask = y_train == label
		test_mask = y_test == label

		plt.scatter(
			x_train[train_mask, 0],
			x_train[train_mask, 1],
			c=colors[idx % len(colors)],
			marker=markers[idx % len(markers)],
			edgecolors="none",
			alpha=0.65,
			label=f"Train class {label}",
		)
		plt.scatter(
			x_test[test_mask, 0],
			x_test[test_mask, 1],
			c=colors[idx % len(colors)],
			marker="x",
			alpha=0.95,
			label=f"Test class {label}",
		)

	plt.title("Data Distribution After Standardization")
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.legend(ncol=2, fontsize=8)
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.show()


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
	plt.figure(figsize=(6.5, 5.5))
	plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()

	positions = np.arange(len(labels))
	plt.xticks(positions, labels)
	plt.yticks(positions, labels)
	plt.xlabel("Predicted label")
	plt.ylabel("True label")

	threshold = cm.max() / 2.0 if cm.size else 0.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			color = "white" if cm[i, j] > threshold else "black"
			plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

	plt.tight_layout()
	plt.show()


def plot_decision_boundary(model, x, y, labels, title="Decision Boundary"):
	x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
	y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8

	xx, yy = np.meshgrid(
		np.linspace(x_min, x_max, 250),
		np.linspace(y_min, y_max, 250),
	)

	grid_points = np.c_[xx.ravel(), yy.ravel()]
	z = model.predict(grid_points).reshape(xx.shape)

	plt.figure(figsize=(8, 6))
	plt.contourf(xx, yy, z, alpha=0.25, levels=np.arange(len(labels) + 1) - 0.5, cmap="Accent")

	markers = ["o", "s", "^"]
	colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown"]
	for idx, label in enumerate(labels):
		mask = y == label
		plt.scatter(
			x[mask, 0],
			x[mask, 1],
			c=colors[idx % len(colors)],
			marker=markers[idx % len(markers)],
			edgecolors="k",
			linewidths=0.3,
			alpha=0.85,
			label=f"Class {label}",
		)

	plt.title(title)
	plt.xlabel("Feature 1")
	plt.ylabel("Feature 2")
	plt.legend()
	plt.grid(alpha=0.2)
	plt.tight_layout()
	plt.show()


def load_iris_dataset():
	"""加载鸢尾花数据集"""
	from sklearn.datasets import load_iris
	iris = load_iris()
	return iris.data[:, :2], iris.target  # 仅使用前两个特征便于可视化


def load_mnist_subset(num_classes=3, samples_per_class=300):
	"""加载 MNIST 数据集的子集"""
	try:
		from sklearn.datasets import load_digits
		digits = load_digits()
		x = digits.data
		y = digits.target
		
		# 只保留前 num_classes 个类别
		mask = y < num_classes
		x = x[mask]
		y = y[mask]
		
		# 随机采样 samples_per_class 个样本
		np.random.seed(42)
		idx = np.random.choice(len(y), min(len(y), num_classes * samples_per_class), replace=False)
		return x[idx], y[idx]
	except:
		print("无法加载 MNIST 数据集，使用人工生成数据")
		return None, None


class GridSearchCV:
	"""网格搜索参数优化"""
	def __init__(self, param_grid):
		self.param_grid = param_grid
		self.best_params = None
		self.best_score = -np.inf
		self.results = []
	
	def fit(self, ModelClass, x_train, y_train, x_val, y_val, strategy="ovo"):
		"""
		进行网格搜索
		
		尝试所有参数组合，在验证集上选择最佳参数
		"""
		# 生成所有参数组合
		param_names = list(self.param_grid.keys())
		param_values = [self.param_grid[name] for name in param_names]
		
		for param_combo in product(*param_values):
			params = dict(zip(param_names, param_combo))
			
			# 创建模型
			if strategy == "ovo":
				model = ModelClass(
					kernel=params.get('kernel'),
					learning_rate=params.get('learning_rate', 0.01),
					lambda_reg=params.get('lambda_reg', 0.01),
					epochs=params.get('epochs', 300)
				)
			else:
				model = ModelClass(
					kernel=params.get('kernel'),
					learning_rate=params.get('learning_rate', 0.01),
					lambda_reg=params.get('lambda_reg', 0.01),
					epochs=params.get('epochs', 300)
				)
			
			# 训练模型
			model.fit(x_train, y_train)
			
			# 验证
			y_pred = model.predict(x_val)
			score = np.mean(y_pred == y_val)
			
			self.results.append({
				'params': params,
				'score': score
			})
			
			print(f"参数: {params} -> 验证准确率: {score:.4f}")
			
			if score > self.best_score:
				self.best_score = score
				self.best_params = params
		
		print(f"\n最优参数: {self.best_params}, 最优分数: {self.best_score:.4f}")
		return self


def generate_multiclass_data(samples_per_class=80, random_state=42):
	rng = np.random.default_rng(random_state)

	centers = np.array([
		[-2.0, -1.5],
		[2.0, 0.0],
		[0.0, 2.5],
	])
	std = 0.9

	x_list = []
	y_list = []
	for class_id, center in enumerate(centers):
		x_class = rng.normal(loc=center, scale=std, size=(samples_per_class, 2))
		y_class = np.full(samples_per_class, class_id)
		x_list.append(x_class)
		y_list.append(y_class)

	x = np.vstack(x_list)
	y = np.concatenate(y_list)
	return x, y


def main():
	"""主程序：演示支持向量机的多分类实现"""
	
	print("=" * 70)
	print("支持向量机多分类实现 - 核函数与参数优化演示")
	print("=" * 70)
	
	# ===== 1) 数据集选择 =====
	print("\n[1] 数据集选择")
	use_iris = True  # 改为 False 使用人工数据或 MNIST
	
	if use_iris:
		print("  使用鸢尾花数据集 (Iris Dataset) - 3 分类问题")
		x, y = load_iris_dataset()
		dataset_name = "Iris"
	else:
		print("  使用人工生成数据 - 3 分类问题")
		x, y = generate_multiclass_data(samples_per_class=100, random_state=7)
		dataset_name = "Synthetic"
	
	# ===== 2) 数据处理 =====
	print(f"\n[2] 数据预处理")
	print(f"  数据集大小: {len(y)} 样本, {x.shape[1]} 特征")
	
	x_train, x_test, y_train, y_test = train_test_split_custom(
		x, y, test_size=0.3, random_state=7
	)
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	print(f"  训练集: {len(y_train)}, 测试集: {len(y_test)}")
	
	# ===== 3) 核函数比较 =====
	print(f"\n[3] 核函数选择与比较")
	kernels = {
		'Linear': LinearKernel(),
		'Polynomial': PolynomialKernel(degree=2),
		'Gaussian': GaussianKernel(gamma=0.1)
	}
	
	results = {}
	for kernel_name, kernel in kernels.items():
		print(f"\n  测试 {kernel_name} 核函数:")
		print(f"    数学形式:", end="")
		if kernel_name == 'Linear':
			print(f" κ(x_i, x_j) = x_i^T x_j")
		elif kernel_name == 'Polynomial':
			print(f" κ(x_i, x_j) = (x_i^T x_j + 1)^d")
		elif kernel_name == 'Gaussian':
			print(f" κ(x_i, x_j) = exp(-γ||x_i - x_j||^2)")
		
		# 训练 OvO 模型
		model = OneVsOneSVM(
			kernel=kernel,
			learning_rate=0.01,
			lambda_reg=0.01,
			epochs=300
		)
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		acc = np.mean(y_pred == y_test)
		print(f"    测试准确率: {acc:.4f}")
		results[kernel_name] = {'model': model, 'accuracy': acc, 'y_pred': y_pred}
	
	# ===== 4) 多分类策略比较 =====
	print(f"\n[4] 多分类策略比较 (使用 Gaussian 核)")
	strategies = {
		'One-vs-Rest': OneVsRestSVM(kernel=GaussianKernel(gamma=0.1), learning_rate=0.01, lambda_reg=0.01, epochs=300),
		'One-vs-One': OneVsOneSVM(kernel=GaussianKernel(gamma=0.1), learning_rate=0.01, lambda_reg=0.01, epochs=300)
	}
	
	for strategy_name, model in strategies.items():
		print(f"\n  {strategy_name}:")
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		acc = np.mean(y_pred == y_test)
		
		# 计算混淆矩阵
		labels = np.unique(y)
		cm = confusion_matrix(y_test, y_pred, labels)
		f1_macro = macro_f1_from_confusion_matrix(cm)
		
		print(f"    准确率: {acc:.4f}")
		print(f"    Macro-F1: {f1_macro:.4f}")
		print(f"    计算复杂度: {len(strategies[strategy_name].pair_models_ if hasattr(strategies[strategy_name], 'pair_models_') else strategies[strategy_name].models_)} 个二分类器")
	
	# ===== 5) 网格搜索参数优化 =====
	print(f"\n[5] 网格搜索参数优化 (样本数据)")
	print(f"  搜索空间: C=(λ) ∈ {{0.001, 0.01, 0.1}}, γ ∈ {{0.01, 0.1}}")
	
	# 从训练集中划分验证集
	x_train_split, x_val, y_train_split, y_val = train_test_split_custom(
		x_train, y_train, test_size=0.2, random_state=42
	)
	
	param_grid = {
		'kernel': [LinearKernel(), GaussianKernel(gamma=0.01), GaussianKernel(gamma=0.1)],
		'lambda_reg': [0.001, 0.01, 0.1],
		'epochs': [200]
	}
	
	gs = GridSearchCV(param_grid)
	gs.fit(OneVsOneSVM, x_train_split, y_train_split, x_val, y_val, strategy="ovo")
	
	# ===== 6) 最终模型评估 =====
	print(f"\n[6] 最终模型评估 (最优参数)")
	best_model = OneVsOneSVM(
		kernel=results['Gaussian']['model'].pair_models_[list(results['Gaussian']['model'].pair_models_.keys())[0]].kernel,
		learning_rate=0.01,
		lambda_reg=0.01,
		epochs=300
	)
	best_model.fit(x_train, y_train)
	y_pred_final = best_model.predict(x_test)
	acc_final = np.mean(y_pred_final == y_test)
	
	labels = np.unique(y)
	cm_final = confusion_matrix(y_test, y_pred_final, labels)
	f1_final = macro_f1_from_confusion_matrix(cm_final)
	
	print(f"  最终测试准确率: {acc_final:.4f}")
	print(f"  最终 Macro-F1: {f1_final:.4f}")
	print(f"  混淆矩阵:")
	print(cm_final)
	
	print("\n" + "=" * 70)


if __name__ == "__main__":
	main()
