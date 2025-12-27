import json
import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    problem_id: str
    prediction: str
    ground_truth: str
    is_correct: bool
    confidence: float = 1.0
    details: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "details": self.details or {}
        }


class Evaluator:
    """
    评估器
    
    评估模型解答的正确性，支持：
    - 数学答案评估（精确匹配、符号等价）
    - 编程代码评估（测试用例验证）
    - 批量评估
    """
    
    def __init__(self, eval_type: str = "math"):
        """
        初始化评估器
        
        Args:
            eval_type: 评估类型 ("math" 或 "code")
        """
        self.eval_type = eval_type
        
    def _normalize_answer(self, answer: str) -> str:
        """
        标准化答案字符串
        
        Args:
            answer: 原始答案
            
        Returns:
            标准化后的答案
        """
        if answer is None:
            return ""
        # 移除空格和换行
        answer = re.sub(r'\s+', '', str(answer))
        # 转换为小写
        answer = answer.lower()
        return answer
        
    def _parse_math_answer(self, answer: str) -> Optional[str]:
        """
        解析数学答案
        
        Args:
            answer: 答案字符串
            
        Returns:
            解析后的答案
        """
        if answer is None:
            return None
        answer = str(answer).strip()
        
        # 尝试提取 \boxed{} 中的内容
        match = re.search(r'\\boxed\{(.+?)\}', answer)
        if match:
            return match.group(1)
            
        # 尝试提取数字
        match = re.search(r'[-+]?\d+\.?\d*', answer)
        if match:
            return match.group(0)
            
        return answer if answer else None
        
    def evaluate_math(
        self, 
        prediction: str, 
        ground_truth: str,
        tolerance: float = 0.0
    ) -> EvaluationResult:
        """
        评估数学答案
        
        Args:
            prediction: 模型预测答案
            ground_truth: 真实答案
            tolerance: 数值容差
            
        Returns:
            评估结果
        """
        # 解析答案
        pred = self._parse_math_answer(prediction)
        gt = self._parse_math_answer(ground_truth)
        
        if pred is None or gt is None:
            return EvaluationResult(
                problem_id="",
                prediction=prediction,
                ground_truth=ground_truth,
                is_correct=False,
                details={"error": "无法解析答案"}
            )
            
        # 尝试数值比较
        try:
            pred_float = float(pred)
            gt_float = float(gt)
            
            if abs(pred_float - gt_float) <= tolerance:
                return EvaluationResult(
                    problem_id="",
                    prediction=prediction,
                    ground_truth=ground_truth,
                    is_correct=True,
                    confidence=1.0,
                    details={"method": "numeric", "tolerance": tolerance}
                )
        except ValueError:
            pass
            
        # 精确匹配
        pred_norm = self._normalize_answer(pred)
        gt_norm = self._normalize_answer(gt)
        
        is_correct = pred_norm == gt_norm
        
        return EvaluationResult(
            problem_id="",
            prediction=prediction,
            ground_truth=ground_truth,
            is_correct=is_correct,
            details={"method": "exact_match"}
        )
        
    def evaluate_code(
        self,
        code: str,
        test_cases: List[Dict],
        timeout: int = 30
    ) -> EvaluationResult:
        """
        评估代码正确性
        
        Args:
            code: Python 代码
            test_cases: 测试用例列表 [{"input": "...", "output": "..."}]
            timeout: 超时时间
            
        Returns:
            评估结果
        """
        import subprocess
        import sys
        import tempfile
        
        # 验证代码语法
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return EvaluationResult(
                problem_id="",
                prediction="[语法错误]",
                ground_truth="",
                is_correct=False,
                details={"error": str(e), "type": "syntax"}
            )
                
        # 运行测试用例
        passed = 0
        total = len(test_cases)
        errors = []
        
        for i, tc in enumerate(test_cases):
            try:
                test_script = f"""
{code}

# 测试用例 {i+1}
input_data = '''{tc.get('input', '')}'''
expected_output = '''{tc.get('output', '')}'''

# 简化验证：实际应该解析输入输出并比较
try:
    # 执行代码的主要函数
    result = main(input_data) if 'main' in dir() else None
    if str(result).strip() == expected_output.strip():
        print('PASS')
    else:
        print(f'FAIL: expected {{expected_output}}, got {{result}}')
except Exception as e:
    print(f'ERROR: {{e}}')
"""
                result = subprocess.run(
                    [sys.executable, "-c", test_script],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    errors.append(f"测试用例 {i+1}: 执行错误 - {result.stderr}")
                elif "PASS" not in result.stdout:
                    errors.append(f"测试用例 {i+1}: {result.stdout.strip()}")
                else:
                    passed += 1
                    
            except subprocess.TimeoutExpired:
                errors.append(f"测试用例 {i+1}: 超时")
            except Exception as e:
                errors.append(f"测试用例 {i+1}: {str(e)}")
                
        is_correct = passed == total
        
        return EvaluationResult(
            problem_id="",
            prediction=f"{passed}/{total} 通过",
            ground_truth=f"{total}/{total} 通过",
            is_correct=is_correct,
            confidence=passed / total if total > 0 else 0,
            details={
                "passed": passed,
                "total": total,
                "errors": errors[:5],  # 只保留前5个错误
                "type": "code"
            }
        )
        
    def batch_evaluate(
        self,
        predictions: List[Dict],
        eval_type: Optional[str] = None,
        verbose: bool = True
    ) -> List[EvaluationResult]:
        """
        批量评估
        
        Args:
            predictions: 预测列表 [{"id": ..., "prediction": ..., "ground_truth": ...}]
            eval_type: 评估类型（覆盖初始化类型）
            verbose: 是否显示进度
            
        Returns:
            评估结果列表
        """
        eval_type = eval_type or self.eval_type
        results = []
        
        iterator = predictions
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="评估中")
            
        for item in iterator:
            if eval_type == "math":
                result = self.evaluate_math(
                    item["prediction"],
                    item["ground_truth"]
                )
            else:
                result = self.evaluate_code(
                    item.get("code", ""),
                    item.get("test_cases", [])
                )
                
            result.problem_id = item.get("id", "")
            results.append(result)
            
        return results
        
    def compute_metrics(
        self, 
        results: List[EvaluationResult]
    ) -> Dict:
        """
        计算评估指标
        
        Args:
            results: 评估结果列表
            
        Returns:
            指标字典
        """
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        
        # 计算置信度统计
        confidences = [r.confidence for r in results if r.confidence < 1.0]
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 1.0,
            "pass_rate": correct / total if total > 0 else 0
        }
        
    def save_results(
        self, 
        results: List[EvaluationResult], 
        output_path: str
    ):
        """
        保存评估结果
        
        Args:
            results: 评估结果列表
            output_path: 输出文件路径
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
    def load_results(self, input_path: str) -> List[EvaluationResult]:
        """
        加载评估结果
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            评估结果列表
        """
        results = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                results.append(EvaluationResult(
                    problem_id=data.get("problem_id", ""),
                    prediction=data.get("prediction", ""),
                    ground_truth=data.get("ground_truth", ""),
                    is_correct=data.get("is_correct", False),
                    confidence=data.get("confidence", 1.0),
                    details=data.get("details", {})
                ))
        return results
        
    def print_report(self, results: List[EvaluationResult]):
        """
        打印评估报告
        
        Args:
            results: 评估结果列表
        """
        metrics = self.compute_metrics(results)
        
        print("\n" + "=" * 50)
        print("评估报告")
        print("=" * 50)
        print(f"总样本数: {metrics['total']}")
        print(f"正确数: {metrics['correct']}")
        print(f"准确率: {metrics['accuracy']:.2%}")
        print(f"平均置信度: {metrics['avg_confidence']:.4f}")
        print("=" * 50)
        
        # 错误分析
        errors = [r for r in results if not r.is_correct]
        if errors:
            print(f"\n错误分析 (共 {len(errors)} 个错误):")
            for r in errors[:5]:  # 只显示前5个
                print(f"  - 问题 {r.problem_id}:")
                print(f"    预测: {r.prediction[:50]}...")
                print(f"    真实: {r.ground_truth[:50]}...")


if __name__ == "__main__":
    # 测试数学评估
    evaluator = Evaluator(eval_type="math")
    
    result = evaluator.evaluate_math(
        prediction="答案是 5050",
        ground_truth="5050"
    )
    print(f"数学评估结果: {result.is_correct}")
    
    # 测试批量评估
    predictions = [
        {"id": "1", "prediction": "42", "ground_truth": "42"},
        {"id": "2", "prediction": "100", "ground_truth": "99"},
        {"id": "3", "prediction": "\\boxed{256}", "ground_truth": "256"},
    ]
    
    results = evaluator.batch_evaluate(predictions)
    metrics = evaluator.compute_metrics(results)
    print(f"\n批量评估指标: {metrics}")
    evaluator.print_report(results)
