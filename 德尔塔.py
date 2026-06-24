import pandas as pd
from rdkit import Chem
import os
from tqdm import tqdm
import gc


# 从给定的代码中提取核心计算逻辑
class NewParameterCalculator:
    """
    基于三阶组贡献法计算新参数的工具类
    分别处理一阶、二阶、三阶基团，并计算新参数值
    """

    def __init__(self, baseline_value: float = 21.6654):
        # 尝试导入groupy模块
        try:
            from groupy.gp_loader import Loader
            from groupy.gp_counter import Counter
            self.loader = Loader()
            self.counter = Counter()

            # 加载所有参数
            self.all_parameters = self.loader.load_parameters(parameter_type='step_wise', split=False)

            # 权重系数（根据公式，w和z均为1）
            self.w = 1.0  # 二阶权重
            self.z = 1.0  # 三阶权重

            # 基准值
            self.baseline_value = baseline_value

            # 根据基团ID前缀分离参数
            self.parameters_1st = {}
            self.parameters_2nd = {}
            self.parameters_3rd = {}

            self._separate_parameters()

            # 预计算基团参数查找表，避免每次计算都进行字符串比较
            self._build_parameter_lookup()

        except ImportError as e:
            print(f"错误: 无法导入groupy模块: {e}")
            print("请确保groupy已安装: pip install groupy")
            raise

    def _separate_parameters(self):
        """
        根据基团ID前缀将参数分离到不同的阶数中
        """
        for group_id, params in self.all_parameters.items():
            if isinstance(group_id, str):
                if group_id.startswith('f_'):
                    self.parameters_1st[group_id] = params
                elif group_id.startswith('s_'):
                    self.parameters_2nd[group_id] = params
                elif group_id.startswith('t_'):
                    self.parameters_3rd[group_id] = params
                else:
                    # 如果基团ID不符合前缀规则，默认视为一阶基团
                    self.parameters_1st[group_id] = params
            else:
                # 如果group_id不是字符串，默认视为一阶基团
                self.parameters_1st[group_id] = params

    def _build_parameter_lookup(self):
        """构建参数查找表，加速参数获取"""
        self.param_lookup = {}

        # 合并所有参数
        for group_id, params in self.parameters_1st.items():
            self.param_lookup[group_id] = ('1st', params.get('δ', 0.0))

        for group_id, params in self.parameters_2nd.items():
            self.param_lookup[group_id] = ('2nd', params.get('δ', 0.0))

        for group_id, params in self.parameters_3rd.items():
            self.param_lookup[group_id] = ('3rd', params.get('δ', 0.0))

    def calculate_delta(self, smiles: str):
        """
        计算分子的δ参数值（优化版）
        """
        if not smiles or pd.isna(smiles) or str(smiles).strip() == '':
            return None

        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None

            # 获取基团计数
            group_info = self.counter.count_a_mol(mol, clear_mode=True, add_note=True)
            group_counts = {k: v for k, v in group_info.items() if k != 'note'}

            # 初始化新参数值（从基准值开始）
            delta_value = self.baseline_value

            # 使用预构建的查找表加速
            for group_id, count in group_counts.items():
                if group_id in self.param_lookup:
                    group_type, delta_param = self.param_lookup[group_id]

                    if group_type == '1st':
                        delta_value += count * delta_param
                    elif group_type == '2nd':
                        delta_value += self.w * count * delta_param
                    else:  # '3rd'
                        delta_value += self.z * count * delta_param

            return round(delta_value, 4)

        except Exception:
            return None


# 主程序
if __name__ == "__main__":
    # 文件路径
    input_file = r"E:\Python\pythonProject\new_t_predict\data\合理分子.csv"
    output_file = r"E:\Python\pythonProject\new_t_predict\data\合理分子_delta.csv"

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        exit()

    print(f"读取文件: {input_file}")
    print("初始化δ参数计算器...")

    try:
        # 初始化计算器
        calculator = NewParameterCalculator(baseline_value=21.6654)
    except Exception as e:
        print(f"初始化失败: {e}")
        exit()

    # 获取文件总行数
    print("正在获取文件总行数...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            header_line = f.readline()
            total_rows = sum(1 for line in f)
    except UnicodeDecodeError:
        try:
            with open(input_file, 'r', encoding='gbk') as f:
                header_line = f.readline()
                total_rows = sum(1 for line in f)
        except:
            print("无法读取文件")
            exit()

    print(f"总数据行数: {total_rows:,}")

    # 读取数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except:
            print("无法读取文件")
            exit()

    # 检查是否有solute列
    if 'solute' not in df.columns:
        print("错误: 文件中没有 'solute' 列")
        print(f"可用列名: {list(df.columns)}")
        exit()

    print(f"开始计算δ参数...")

    # 计算δ参数并显示进度条
    delta_values = []
    for smiles in tqdm(df['solute'], total=len(df), desc="计算δ参数"):
        delta = calculator.calculate_delta(smiles)
        delta_values.append(delta)

    # 将δ参数添加到DataFrame
    df['δ'] = delta_values

    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n计算完成!")
    print(f"结果已保存到: {output_file}")
    print(f"前5行结果预览:")
    print(df[['solute', 'δ']].head())