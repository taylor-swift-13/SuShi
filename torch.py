"""
PyTorch 常用 API 详细讲解（附示例与测试代码）
================================================
直接运行: python torch_api_guide.py
所有测试通过会打印 ✅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 一、张量创建
# ============================================================

def demo_tensor_creation():
    """1.1 基本创建"""
    # 从 Python 列表创建
    a = torch.tensor([1, 2, 3])                          # 一维，int64
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])           # 二维，float32
    c = torch.tensor([1, 2, 3], dtype=torch.float32)     # 强制 float32

    print(f"a = {a}, dtype={a.dtype}")
    print(f"b shape={b.shape}, dtype={b.dtype}")
    print(f"c dtype={c.dtype}")


def demo_special_tensors():
    """1.2 特殊张量"""
    z = torch.zeros(2, 3)          # 全 0
    o = torch.ones(3, 4)           # 全 1
    f = torch.full((2, 2), 7.0)    # 全 7.0
    eye = torch.eye(3)             # 单位矩阵

    r = torch.arange(0, 10, 2)         # tensor([0, 2, 4, 6, 8])
    l = torch.linspace(0, 1, 5)        # tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    rand_uniform = torch.rand(2, 3)           # [0, 1) 均匀分布
    rand_normal  = torch.randn(2, 3)          # 标准正态 N(0, 1)
    rand_int     = torch.randint(0, 10, (2, 3))  # [0, 10) 整数

    print(f"zeros:\n{z}")
    print(f"arange: {r}")
    print(f"linspace: {l}")


def demo_like():
    """1.3 *_like 系列"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.zeros_like(x)    # shape 和 dtype 跟 x 一样，值全 0
    o = torch.ones_like(x)
    r = torch.randn_like(x)
    print(f"zeros_like: {z}")


def test_tensor_creation():
    a = torch.tensor([1, 2, 3])
    assert a.shape == (3,)
    assert a.dtype == torch.int64

    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert b.shape == (2, 2)
    assert b.dtype == torch.float32

    c = torch.tensor([1, 2, 3], dtype=torch.float16)
    assert c.dtype == torch.float16
    print("✅ test_tensor_creation passed")


def test_special_tensors():
    z = torch.zeros(2, 3)
    assert z.shape == (2, 3)
    assert (z == 0).all()

    o = torch.ones(3, 4)
    assert (o == 1).all()

    f = torch.full((2, 2), 7.0)
    assert (f == 7.0).all()

    r = torch.arange(0, 10, 2)
    assert torch.equal(r, torch.tensor([0, 2, 4, 6, 8]))

    l = torch.linspace(0, 1, 5)
    assert l.shape == (5,)
    assert torch.isclose(l[0], torch.tensor(0.0))
    assert torch.isclose(l[-1], torch.tensor(1.0))
    print("✅ test_special_tensors passed")


# ============================================================
# 二、形状操作
# ============================================================

def demo_reshape():
    """2.1 reshape / view"""
    x = torch.arange(12)        # shape (12,)
    a = x.reshape(3, 4)         # shape (3, 4)
    b = x.reshape(2, -1)        # -1 自动推断 → shape (2, 6)
    c = x.view(4, 3)            # view 要求内存连续

    # view vs reshape：view 要求 contiguous，reshape 自动处理
    print(f"原始: {x.shape}")
    print(f"reshape(3,4): {a.shape}")
    print(f"reshape(2,-1): {b.shape}")


def demo_squeeze_unsqueeze():
    """2.2 squeeze / unsqueeze"""
    x = torch.zeros(1, 3, 1, 4)

    a = x.squeeze()       # shape (3, 4)  —— 去掉所有大小为1的维度
    b = x.squeeze(0)      # shape (3, 1, 4)
    c = x.squeeze(2)      # shape (1, 3, 4)

    y = torch.zeros(3, 4)
    d = y.unsqueeze(0)    # shape (1, 3, 4)
    e = y.unsqueeze(-1)   # shape (3, 4, 1)

    print(f"squeeze(): {a.shape}")
    print(f"unsqueeze(0): {d.shape}")
    print(f"unsqueeze(-1): {e.shape}")


def demo_permute_transpose():
    """2.3 permute / transpose"""
    x = torch.randn(2, 3, 4)

    a = x.transpose(0, 2)    # shape (4, 3, 2)  —— 交换两个维度
    b = x.permute(2, 0, 1)   # shape (4, 2, 3)  —— 任意重排

    m = torch.randn(3, 4)
    print(f".T: {m.T.shape}")  # (4, 3)  —— 只用于 2D


def demo_cat_stack():
    """2.4 拼接与拆分"""
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    c0 = torch.cat([a, b], dim=0)     # shape (4, 2)  竖着拼
    c1 = torch.cat([a, b], dim=1)     # shape (2, 4)  横着拼
    s  = torch.stack([a, b], dim=0)   # shape (2, 2, 2) 新维度

    parts = torch.chunk(c0, 2, dim=0)           # 均匀切 2 份
    p1, p2, p3 = torch.split(torch.arange(10), [3, 3, 4])  # 按大小切

    print(f"cat dim=0: {c0.shape}")
    print(f"cat dim=1: {c1.shape}")
    print(f"stack dim=0: {s.shape}")


def test_reshape():
    x = torch.arange(12)
    a = x.reshape(3, 4)
    assert a.shape == (3, 4)
    assert a[0, 0] == 0
    assert a[2, 3] == 11

    b = x.reshape(2, -1)
    assert b.shape == (2, 6)

    c = x.view(3, 4)
    assert torch.equal(a, c)
    print("✅ test_reshape passed")


def test_squeeze_unsqueeze():
    x = torch.zeros(1, 3, 1, 4)
    assert x.squeeze().shape == (3, 4)
    assert x.squeeze(0).shape == (3, 1, 4)

    y = torch.zeros(3, 4)
    assert y.unsqueeze(0).shape == (1, 3, 4)
    assert y.unsqueeze(-1).shape == (3, 4, 1)

    z = y.unsqueeze(0).squeeze(0)
    assert z.shape == y.shape
    print("✅ test_squeeze_unsqueeze passed")


def test_permute_transpose():
    x = torch.randn(2, 3, 4)

    a = x.transpose(0, 2)
    assert a.shape == (4, 3, 2)
    assert a[0, 0, 0] == x[0, 0, 0]
    assert a[3, 2, 1] == x[1, 2, 3]

    b = x.permute(2, 0, 1)
    assert b.shape == (4, 2, 3)

    m = torch.randn(3, 4)
    assert m.T.shape == (4, 3)
    print("✅ test_permute_transpose passed")


def test_cat_stack():
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    c0 = torch.cat([a, b], dim=0)
    assert c0.shape == (4, 2)
    assert torch.equal(c0, torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]))

    c1 = torch.cat([a, b], dim=1)
    assert c1.shape == (2, 4)

    s = torch.stack([a, b], dim=0)
    assert s.shape == (2, 2, 2)
    assert torch.equal(s[0], a)
    assert torch.equal(s[1], b)

    parts = torch.chunk(c0, 2, dim=0)
    assert len(parts) == 2
    assert torch.equal(parts[0], a)
    print("✅ test_cat_stack passed")


# ============================================================
# 三、索引与切片
# ============================================================

def demo_indexing():
    """3.1 基本索引"""
    x = torch.tensor([[10, 20, 30],
                       [40, 50, 60],
                       [70, 80, 90]])

    print(x[0])         # tensor([10, 20, 30]) —— 第 0 行
    print(x[1, 2])      # tensor(60)
    print(x[:, 1])      # tensor([20, 50, 80]) —— 所有行第 1 列
    print(x[0:2, 1:])   # tensor([[20, 30], [50, 60]])


def demo_fancy_indexing():
    """3.2 高级索引（Fancy Indexing）"""
    x = torch.tensor([[10, 20, 30],
                       [40, 50, 60],
                       [70, 80, 90]])

    # 用张量选取特定位置
    rows = torch.tensor([0, 1, 2])
    cols = torch.tensor([2, 0, 1])
    print(x[rows, cols])  # tensor([30, 40, 80]) —— 取 (0,2), (1,0), (2,1)
    # 这就是交叉熵中 log_softmax[range(batch_size), targets] 的原理

    # 布尔索引
    mask = x > 50
    print(f"mask:\n{mask}")
    print(f"x[mask]: {x[mask]}")  # tensor([60, 70, 80, 90])


def test_indexing():
    x = torch.tensor([[10, 20, 30],
                       [40, 50, 60],
                       [70, 80, 90]])

    rows = torch.tensor([0, 1, 2])
    cols = torch.tensor([2, 0, 1])
    result = x[rows, cols]
    assert torch.equal(result, torch.tensor([30, 40, 80]))

    mask = x > 50
    assert torch.equal(x[mask], torch.tensor([60, 70, 80, 90]))

    assert x[0:2, 1:].shape == (2, 2)
    print("✅ test_indexing passed")


# ============================================================
# 四、数学运算
# ============================================================

def demo_elementwise():
    """4.1 逐元素运算"""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print(f"a + b = {a + b}")              # tensor([5., 7., 9.])
    print(f"a * b = {a * b}")              # tensor([ 4., 10., 18.]) 逐元素乘
    print(f"a ** 2 = {a ** 2}")            # tensor([1., 4., 9.])
    print(f"sqrt(a) = {torch.sqrt(a)}")
    print(f"exp(a) = {torch.exp(a)}")
    print(f"log(a) = {torch.log(a)}")

    # clamp: 限制范围
    x = torch.tensor([-2.0, 0.5, 3.0, 10.0])
    print(f"clamp(0,5): {torch.clamp(x, min=0, max=5)}")


def demo_reductions():
    """4.2 规约运算"""
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

    print(f"sum(): {x.sum()}")                # 21
    print(f"sum(dim=0): {x.sum(dim=0)}")      # [5, 7, 9] 每列的和
    print(f"sum(dim=1): {x.sum(dim=1)}")      # [6, 15]   每行的和
    print(f"mean(): {x.mean()}")              # 3.5
    print(f"max(dim=1): {x.max(dim=1)}")      # values + indices
    print(f"argmax(dim=1): {x.argmax(dim=1)}")

    # keepdim 的意义：方便广播
    s = x.sum(dim=1, keepdim=True)   # shape (2, 1) 而非 (2,)
    normalized = x / s               # (2, 3) / (2, 1) → 广播
    print(f"keepdim shape: {s.shape}")
    print(f"按行归一化:\n{normalized}")


def demo_matmul():
    """4.3 矩阵运算"""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # 矩阵乘法（三种等价写法）
    c1 = a @ b
    c2 = torch.matmul(a, b)
    c3 = torch.mm(a, b)           # 只能 2D
    print(f"a @ b:\n{c1}")

    # 批量矩阵乘法
    batch_a = torch.randn(5, 3, 4)
    batch_b = torch.randn(5, 4, 2)
    batch_c = torch.bmm(batch_a, batch_b)   # shape (5, 3, 2)
    print(f"bmm shape: {batch_c.shape}")

    # 注意区分：* 逐元素，@ 矩阵乘
    print(f"a * b (逐元素):\n{a * b}")
    print(f"a @ b (矩阵乘):\n{a @ b}")


def test_reductions():
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

    assert x.sum().item() == 21.0
    assert torch.equal(x.sum(dim=0), torch.tensor([5., 7., 9.]))
    assert torch.equal(x.sum(dim=1), torch.tensor([6., 15.]))
    assert torch.isclose(x.mean(), torch.tensor(3.5))
    assert torch.equal(x.argmax(dim=1), torch.tensor([2, 2]))

    s = x.sum(dim=1, keepdim=True)
    assert s.shape == (2, 1)

    normalized = x / s
    row_sums = normalized.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(2))
    print("✅ test_reductions passed")


def test_matmul():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    expected = torch.tensor([[19., 22.], [43., 50.]])
    assert torch.equal(a @ b, expected)
    assert torch.equal(torch.matmul(a, b), expected)
    assert torch.equal(torch.mm(a, b), expected)

    ba = torch.randn(5, 3, 4)
    bb = torch.randn(5, 4, 2)
    bc = torch.bmm(ba, bb)
    assert bc.shape == (5, 3, 2)
    print("✅ test_matmul passed")


# ============================================================
# 五、广播（Broadcasting）
# ============================================================

def demo_broadcasting():
    """从右往左对齐维度，每个维度要么相等，要么其中一个是 1"""

    # 例 1：向量 + 标量
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor(10.0)
    print(f"向量+标量: {a + b}")   # [11, 12, 13]

    # 例 2：矩阵 + 行向量
    m = torch.ones(3, 4)
    v = torch.tensor([1, 2, 3, 4])   # (4,) → (1,4) → (3,4)
    print(f"矩阵+行向量 shape: {(m + v).shape}")

    # 例 3：列向量 + 行向量 → 矩阵
    col = torch.tensor([[1], [2], [3]])   # (3, 1)
    row = torch.tensor([10, 20, 30])      # (1, 3)
    result = col + row                     # (3, 3)
    print(f"列+行:\n{result}")

    # 例 4：交叉熵中的数值稳定技巧
    logits = torch.tensor([[2.0, 1.0, 0.1],
                            [0.5, 2.0, 0.3]])
    max_vals = logits.max(dim=1, keepdim=True).values   # (2, 1)
    shifted = logits - max_vals                          # (2,3) - (2,1) 广播
    print(f"shifted logits:\n{shifted}")


def test_broadcasting():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor(10.0)
    assert torch.equal(a + b, torch.tensor([11., 12., 13.]))

    m = torch.ones(3, 4)
    v = torch.arange(4).float()
    result = m + v
    assert result.shape == (3, 4)
    assert torch.equal(result[0], torch.tensor([1., 2., 3., 4.]))

    col = torch.tensor([[1], [2], [3]])
    row = torch.tensor([10, 20, 30])
    result = col + row
    assert result.shape == (3, 3)
    assert result[0, 0] == 11
    assert result[2, 2] == 33
    print("✅ test_broadcasting passed")


# ============================================================
# 六、激活函数
# ============================================================

def demo_activations():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print(f"ReLU:    {torch.relu(x)}")          # max(0, x)
    print(f"Sigmoid: {torch.sigmoid(x)}")       # 1/(1+exp(-x))，输出 (0,1)
    print(f"Tanh:    {torch.tanh(x)}")          # 输出 (-1,1)
    print(f"SiLU:    {F.silu(x)}")              # x * sigmoid(x)，LLM 常用
    print(f"GELU:    {F.gelu(x)}")              # Transformer 常用

    # Softmax: logits → 概率分布
    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = torch.softmax(logits, dim=0)
    print(f"Softmax: {probs}, sum={probs.sum():.4f}")

    # 手写 SiLU 验证
    def my_silu(x):
        return x * torch.sigmoid(x)

    t = torch.randn(5)
    assert torch.allclose(my_silu(t), F.silu(t))
    print("✅ 手写 SiLU 与 PyTorch 一致")


def test_activations():
    x = torch.tensor([-1.0, 0.0, 1.0])

    r = torch.relu(x)
    assert torch.equal(r, torch.tensor([0., 0., 1.]))

    s = torch.sigmoid(x)
    assert (s > 0).all() and (s < 1).all()
    assert torch.isclose(s[1], torch.tensor(0.5))    # sigmoid(0) = 0.5

    logits = torch.tensor([2.0, 1.0, 0.1])
    probs = torch.softmax(logits, dim=0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert probs.argmax() == 0
    print("✅ test_activations passed")


# ============================================================
# 七、自动求导（Autograd）
# ============================================================

def demo_autograd_basic():
    """7.1 基本用法"""
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1     # y = x² + 2x + 1
    y.backward()
    print(f"dy/dx at x=3: {x.grad}")   # 2*3+2 = 8


def demo_autograd_matrix():
    """7.2 矩阵情况"""
    W = torch.randn(3, 2, requires_grad=True)
    x = torch.randn(2, 1)
    y = W @ x
    loss = y.sum()
    loss.backward()
    print(f"W.grad shape: {W.grad.shape}")   # (3, 2)


def demo_grad_accumulation():
    """7.3 梯度累积与清零"""
    x = torch.tensor(2.0, requires_grad=True)

    y1 = x ** 2
    y1.backward()
    print(f"第一次: {x.grad}")       # 4.0

    y2 = x ** 3
    y2.backward()
    print(f"累加后: {x.grad}")       # 4.0 + 12.0 = 16.0

    x.grad.zero_()
    y3 = x ** 2
    y3.backward()
    print(f"清零后: {x.grad}")       # 4.0


def demo_no_grad():
    """7.4 no_grad / detach"""
    x = torch.tensor(3.0, requires_grad=True)

    with torch.no_grad():
        y = x * 2
        print(f"no_grad 内 requires_grad: {y.requires_grad}")   # False

    z = (x * 2).detach()
    print(f"detach 后 requires_grad: {z.requires_grad}")         # False


def test_autograd():
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y.backward()
    assert torch.isclose(x.grad, torch.tensor(8.0))

    x.grad.zero_()
    y2 = x ** 3
    y2.backward()
    assert torch.isclose(x.grad, torch.tensor(27.0))   # 3x² = 27

    with torch.no_grad():
        z = x * 2
        assert not z.requires_grad
    print("✅ test_autograd passed")


# ============================================================
# 八、nn.Module 与自定义层
# ============================================================

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def demo_nn_module():
    """8.1 基本结构"""
    model = SimpleNet(10, 32, 5)
    x = torch.randn(4, 10)
    output = model(x)
    print(f"输出 shape: {output.shape}")   # (4, 5)

    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")


def demo_linear_internals():
    """8.2 nn.Linear 内部原理: y = xW^T + b"""
    linear = nn.Linear(3, 2)
    print(f"weight: {linear.weight.shape}")   # (2, 3) 注意是转置的
    print(f"bias: {linear.bias.shape}")       # (2,)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    manual = x @ linear.weight.T + linear.bias
    auto   = linear(x)
    assert torch.allclose(manual, auto)
    print("✅ nn.Linear 手动 vs 自动一致")


def demo_common_layers():
    """8.3 常用层"""
    # LayerNorm
    ln = nn.LayerNorm(4)
    x = torch.randn(2, 3, 4)
    out = ln(x)
    print(f"LayerNorm 输出 shape: {out.shape}")

    # Embedding
    emb = nn.Embedding(1000, 64)
    token_ids = torch.tensor([5, 10, 100])
    vectors = emb(token_ids)
    print(f"Embedding 输出: {vectors.shape}")   # (3, 64)

    # Dropout
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(10)
    print(f"训练模式: {dropout(x)}")
    dropout.eval()
    print(f"评估模式: {dropout(x)}")


def test_nn_modules():
    model = SimpleNet(10, 32, 5)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 5)

    total = sum(p.numel() for p in model.parameters())
    assert total == 10 * 32 + 32 + 32 * 5 + 5   # 517

    ln = nn.LayerNorm(4)
    x = torch.randn(2, 3, 4)
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
    print("✅ test_nn_modules passed")


# ============================================================
# 九、损失函数与优化器
# ============================================================

def demo_loss_functions():
    """9.1 常用损失函数"""
    # 交叉熵（分类）
    logits = torch.tensor([[2.0, 1.0, 0.1],
                            [0.5, 2.0, 0.3]])
    targets = torch.tensor([0, 1])
    loss_ce = F.cross_entropy(logits, targets)
    print(f"交叉熵: {loss_ce:.4f}")

    # MSE（回归）
    pred = torch.tensor([2.5, 0.0, 2.1])
    true = torch.tensor([3.0, -0.5, 2.0])
    loss_mse = F.mse_loss(pred, true)
    print(f"MSE: {loss_mse:.4f}")

    # L1
    loss_l1 = F.l1_loss(pred, true)
    print(f"L1: {loss_l1:.4f}")


def demo_training_loop():
    """9.2 完整训练循环"""
    model = SimpleNet(10, 32, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X = torch.randn(100, 10)
    Y = torch.randint(0, 3, (100,))

    for epoch in range(5):
        optimizer.zero_grad()                      # 1. 清零梯度
        logits = model(X)                          # 2. 前向传播
        loss = F.cross_entropy(logits, Y)          # 3. 计算损失
        loss.backward()                            # 4. 反向传播
        optimizer.step()                           # 5. 更新参数
        print(f"  Epoch {epoch}: loss={loss.item():.4f}")


def test_training_loop():
    model = SimpleNet(10, 32, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    X = torch.randn(50, 10)
    Y = torch.randint(0, 3, (50,))

    with torch.no_grad():
        loss_before = F.cross_entropy(model(X), Y).item()

    for _ in range(50):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X), Y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        loss_after = F.cross_entropy(model(X), Y).item()

    assert loss_after < loss_before
    print(f"✅ test_training_loop passed: {loss_before:.4f} → {loss_after:.4f}")


# ============================================================
# 十、实用技巧
# ============================================================

def demo_type_device():
    """10.1 类型与设备转换"""
    x = torch.tensor([1, 2, 3])
    x_float = x.float()            # float32
    x_half  = x.half()             # float16
    x_int   = x_float.int()        # int32
    x_to    = x.to(torch.float64)  # 通用写法
    print(f"dtypes: {x.dtype}, {x_float.dtype}, {x_half.dtype}, {x_to.dtype}")

    # GPU（需要 CUDA）:
    # x_gpu = x.to('cuda')  /  x.cuda()
    # x_cpu = x_gpu.cpu()


def demo_numpy_interop():
    """10.2 与 NumPy 互转"""
    # Tensor → NumPy（共享内存）
    t = torch.tensor([1.0, 2.0, 3.0])
    n = t.numpy()
    print(f"type: {type(n)}, value: {n}")

    # NumPy → Tensor（共享内存）
    n2 = np.array([4.0, 5.0, 6.0])
    t2 = torch.from_numpy(n2)

    # 断开共享用 .clone()
    t3 = torch.from_numpy(n2).clone()
    n2[0] = 999
    print(f"共享: t2[0]={t2[0]}, 独立: t3[0]={t3[0]}")


def demo_where_masked_fill():
    """10.3 where / masked_fill"""
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])

    # where: 条件选择（相当于 ReLU）
    result = torch.where(x > 0, x, torch.zeros_like(x))
    print(f"where: {result}")

    # masked_fill: attention mask 常用
    scores = torch.randn(2, 4)
    mask = torch.tensor([[1, 1, 0, 0],
                          [1, 1, 1, 0]], dtype=torch.bool)
    masked = scores.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(masked, dim=-1)
    print(f"masked softmax:\n{probs}")


def demo_tril_causal_mask():
    """10.4 因果掩码"""
    seq_len = 4
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"因果掩码:\n{causal_mask}")

    scores = torch.randn(seq_len, seq_len)
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"因果 attention:\n{attn_weights}")


def test_practical():
    # where
    x = torch.tensor([1.0, -2.0, 3.0, -4.0])
    result = torch.where(x > 0, x, torch.zeros_like(x))
    assert torch.equal(result, torch.tensor([1., 0., 3., 0.]))

    # masked_fill + softmax
    scores = torch.ones(3, 3)
    mask = torch.tril(torch.ones(3, 3)).bool()
    masked = scores.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(masked, dim=-1)
    assert torch.allclose(probs[0], torch.tensor([1., 0., 0.]))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))

    causal = torch.tril(torch.ones(4, 4))
    assert causal[0, 1] == 0
    assert causal[1, 0] == 1
    print("✅ test_practical passed")


# ============================================================
# 十一、Einsum（爱因斯坦求和）
# ============================================================

def demo_einsum():
    # 矩阵乘法
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    c = torch.einsum('ij,jk->ik', a, b)
    assert torch.allclose(c, a @ b)

    # 批量矩阵乘法
    a = torch.randn(8, 3, 4)
    b = torch.randn(8, 4, 5)
    c = torch.einsum('bij,bjk->bik', a, b)
    assert torch.allclose(c, torch.bmm(a, b))

    # 向量点积
    a = torch.randn(5)
    b = torch.randn(5)
    c = torch.einsum('i,i->', a, b)
    assert torch.isclose(c, torch.dot(a, b))

    # 转置
    m = torch.randn(3, 4)
    mt = torch.einsum('ij->ji', m)
    assert torch.equal(mt, m.T)

    # 对角线
    m = torch.randn(3, 3)
    d = torch.einsum('ii->i', m)
    assert torch.equal(d, m.diagonal())

    print("✅ einsum 所有示例通过")


# ============================================================
# 主函数：运行所有 demo 和测试
# ============================================================

if __name__ == "__main__":
    sections = [
        ("一、张量创建", [
            demo_tensor_creation, demo_special_tensors, demo_like,
            test_tensor_creation, test_special_tensors,
        ]),
        ("二、形状操作", [
            demo_reshape, demo_squeeze_unsqueeze, demo_permute_transpose, demo_cat_stack,
            test_reshape, test_squeeze_unsqueeze, test_permute_transpose, test_cat_stack,
        ]),
        ("三、索引与切片", [
            demo_indexing, demo_fancy_indexing,
            test_indexing,
        ]),
        ("四、数学运算", [
            demo_elementwise, demo_reductions, demo_matmul,
            test_reductions, test_matmul,
        ]),
        ("五、广播", [
            demo_broadcasting,
            test_broadcasting,
        ]),
        ("六、激活函数", [
            demo_activations,
            test_activations,
        ]),
        ("七、自动求导", [
            demo_autograd_basic, demo_autograd_matrix, demo_grad_accumulation, demo_no_grad,
            test_autograd,
        ]),
        ("八、nn.Module", [
            demo_nn_module, demo_linear_internals, demo_common_layers,
            test_nn_modules,
        ]),
        ("九、损失函数与优化器", [
            demo_loss_functions, demo_training_loop,
            test_training_loop,
        ]),
        ("十、实用技巧", [
            demo_type_device, demo_numpy_interop, demo_where_masked_fill, demo_tril_causal_mask,
            test_practical,
        ]),
        ("十一、Einsum", [
            demo_einsum,
        ]),
    ]

    for title, funcs in sections:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for fn in funcs:
            print(f"\n--- {fn.__name__} ---")
            fn()

    print(f"\n{'='*60}")
    print(" 全部完成！所有测试通过 ✅")
    print(f"{'='*60}")