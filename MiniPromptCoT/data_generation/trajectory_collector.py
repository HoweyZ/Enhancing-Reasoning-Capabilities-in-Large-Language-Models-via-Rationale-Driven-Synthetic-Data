import json
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Trajectory:
    """轨迹数据类"""
    problem_id: str
    problem_text: str
    response: str
    answer: Optional[str] = None
    is_correct: Optional[bool] = None
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem_text,
            "response": self.response,
            "answer": self.answer,
            "is_correct": self.is_correct,
            "metadata": self.metadata or {}
        }


class TrajectoryCollector:
    """
    轨迹收集器
    
    收集 LLM 对问题的解答轨迹，支持：
    - 单次采样
    - 多次采样（用于自我博弈）
    - 思维链 (CoT) 格式
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = True
    ):
        """
        初始化轨迹收集器
        
        Args:
            model_path: 模型路径
            temperature: 生成温度
            max_tokens: 最大生成长度
            enable_thinking: 是否启用思维模式
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=1,
                dtype="half",
                enforce_eager=True
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95
            )
            self._use_vllm = True
        except ImportError:
            self._use_vllm = False
            
        if not self._use_vllm:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
    def _generate_with_llm(self, prompt: str) -> str:
        """使用 LLM 生成内容"""
        if self._use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True
                )
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            return response
            
    def _build_prompt(
        self, 
        problem_text: str,
        task_type: str = "math",
        include_answer_prefix: bool = True
    ) -> str:
        """
        构建问题 prompt
        
        Args:
            problem_text: 问题描述
            task_type: 任务类型 ("math" 或 "code")
            include_answer_prefix: 是否包含答案前缀
            
        Returns:
            格式化的 prompt
        """
        if task_type == "math":
            prompt = f"""请解决以下数学问题。请详细写出推理过程，最后给出答案。

问题：{problem_text}

请按以下格式回答：
1. 详细的推理步骤
2. 最终答案（用 \\boxed{{}} 包裹）

"""
        else:  # code
            prompt = f"""请解决以下编程问题。请写出完整的 Python 代码。

问题：{problem_text}

请按以下格式回答：
1. 算法思路
2. 完整代码（用 ```python ... ``` 包裹）

"""
        return prompt
        
    def _extract_answer(self, response: str, task_type: str = "math") -> Optional[str]:
        """
        从响应中提取答案
        
        Args:
            response: 模型响应
            task_type: 任务类型
            
        Returns:
            提取的答案
        """
        if task_type == "math":
            # 尝试匹配 \boxed{...}
            match = re.search(r'\\boxed\{(.+?)\}', response)
            if match:
                return match.group(1)
            # 尝试匹配 "答案: ..." 格式
            match = re.search(r'[答案|answer|Answer][:：]\s*(.+)', response)
            if match:
                return match.group(1).strip()
        else:
            # 提取代码块
            code_match = re.search(r'```python\s*\n(.+?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
                
        return None
        
    def collect_single(
        self, 
        problem_id: str,
        problem_text: str,
        task_type: str = "math"
    ) -> Trajectory:
        """
        收集单个问题的解答轨迹
        
        Args:
            problem_id: 问题 ID
            problem_text: 问题描述
            task_type: 任务类型
            
        Returns:
            收集的轨迹
        """
        prompt = self._build_prompt(problem_text, task_type)
        response = self._generate_with_llm(prompt)
        answer = self._extract_answer(response, task_type)
        
        return Trajectory(
            problem_id=problem_id,
            problem_text=problem_text,
            response=response,
            answer=answer,
            metadata={"task_type": task_type}
        )
        
    def collect_multiple(
        self,
        problems: List[Dict],
        task_type: str = "math",
        num_samples: int = 8,
        verbose: bool = True
    ) -> List[List[Trajectory]]:
        """
        收集多个问题的多次解答轨迹（用于 Self-Play）
        
        Args:
            problems: 问题列表 (每个元素包含 id 和 problem)
            task_type: 任务类型
            num_samples: 每个问题采样的次数
            verbose: 是否显示进度
            
        Returns:
            每个问题的多次轨迹列表
        """
        all_trajectories = []
        
        if verbose:
            pbar = tqdm(total=len(problems) * num_samples, desc="收集轨迹")
            
        for problem in problems:
            problem_trajectories = []
            for _ in range(num_samples):
                trajectory = self.collect_single(
                    problem["id"],
                    problem["problem"],
                    task_type
                )
                problem_trajectories.append(trajectory)
                all_trajectories.append(trajectory)
                
                if verbose:
                    pbar.update(1)
                    
        if verbose:
            pbar.close()
            
        # 按问题分组
        grouped = []
        for i in range(0, len(all_trajectories), num_samples):
            grouped.append(all_trajectories[i:i + num_samples])
            
        return grouped
        
    def save_trajectories(
        self, 
        trajectories: List[Trajectory], 
        output_path: str
    ):
        """
        保存轨迹到文件
        
        Args:
            trajectories: 轨迹列表
            output_path: 输出文件路径
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for traj in trajectories:
                json.dump(traj.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
    def load_trajectories(self, input_path: str) -> List[Trajectory]:
        """
        从文件加载轨迹
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            轨迹列表
        """
        trajectories = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                trajectories.append(Trajectory(
                    problem_id=data["problem_id"],
                    problem_text=data["problem"],
                    response=data["response"],
                    answer=data.get("answer"),
                    is_correct=data.get("is_correct"),
                    metadata=data.get("metadata", {})
                ))
        return trajectories
        
    def analyze_trajectories(
        self, 
        trajectories: List[Trajectory]
    ) -> Dict:
        """
        分析轨迹数据
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            统计信息字典
        """
        total = len(trajectories)
        with_answer = sum(1 for t in trajectories if t.answer is not None)
        correct = sum(1 for t in trajectories if t.is_correct)
        
        # 响应长度统计
        lengths = [len(t.response) for t in trajectories]
        
        return {
            "total": total,
            "with_answer": with_answer,
            "correct": correct,
            "avg_length": sum(lengths) / total if total > 0 else 0,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "accuracy": correct / total if total > 0 else 0
        }


# 延迟导入 torch
import torch


if __name__ == "__main__":
    # 测试
    collector = TrajectoryCollector(use_vllm=False)
    
    problem = {
        "id": "test_001",
        "problem": "计算 1+2+3+...+100 的值。"
    }
    
    trajectory = collector.collect_single(
        problem["id"],
        problem["problem"],
        "math"
    )
    
    print(f"问题 ID: {trajectory.problem_id}")
    print(f"答案: {trajectory.answer}")
    print(f"响应长度: {len(trajectory.response)}")
