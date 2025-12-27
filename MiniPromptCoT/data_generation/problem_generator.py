import json
import re
from typing import List, Dict, Optional
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Problem:
    """问题数据类"""
    id: str
    concepts: List[str]
    difficulty: str
    problem_text: str
    answer: Optional[str] = None
    solution: Optional[str] = None
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "concepts": self.concepts,
            "difficulty": self.difficulty,
            "problem": self.problem_text,
            "answer": self.answer,
            "solution": self.solution,
            "metadata": self.metadata or {}
        }


class ProblemGenerator:
    """
    问题生成器
    
    使用 LLM 生成数学/编程问题，支持：
    - 单个问题生成
    - 批量问题生成
    - 问题质量过滤
    """
    
    # 问题生成 Prompt 模板
    MATH_PROBLEM_TEMPLATE = """你是一位数学竞赛命题专家。请根据给定的概念和难度，设计一道高质量的数学问题。

【要求】
1. 问题必须清晰、严谨，有唯一正确答案
2. 问题应该能够测试学生对给定概念的掌握程度
3. 难度应与指定级别匹配
4. 问题表述应该简洁明了

【输入】
- 概念: {concepts}
- 难度: {difficulty}

【输出格式】
请按以下格式输出：

```json
{{
    "problem": "问题描述...",
    "answer": "答案...",
    "solution": "简要解题思路..."
}}
```
"""
    
    CODE_PROBLEM_TEMPLATE = """你是一位编程竞赛命题专家。请根据给定的概念和难度，设计一道高质量的编程问题。

【要求】
1. 问题必须有明确的输入输出格式
2. 问题应该能够测试算法设计能力
3. 难度应与指定级别匹配
4. 问题应该有多种可能的解法

【输入】
- 概念: {concepts}
- 难度: {difficulty}

【输出格式】
请按以下格式输出：

```json
{{
    "problem": "问题描述（包括输入输出格式）...",
    "answer": "代码框架...",
    "solution": "算法思路..."
}}
```
"""
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_vllm: bool = True
    ):
        """
        初始化问题生成器
        
        Args:
            model_path: 模型路径
            temperature: 生成温度
            max_tokens: 最大生成长度
            use_vllm: 是否使用 vLLM 加速
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_vllm = use_vllm
        
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        if self.use_vllm:
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
            
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 返回的 JSON 响应
        
        Args:
            response: LLM 原始输出
            
        Returns:
            解析后的字典，失败返回 None
        """
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```json\s*\n(.+?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_str = response
            
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            return None
            
    def generate_problem(
        self, 
        concepts: List[str],
        difficulty: str,
        problem_type: str = "math"
    ) -> Optional[Problem]:
        """
        生成单个问题
        
        Args:
            concepts: 概念列表
            difficulty: 难度级别
            problem_type: 问题类型 ("math" 或 "code")
            
        Returns:
            生成的问题，失败返回 None
        """
        # 选择模板
        if problem_type == "math":
            template = self.MATH_PROBLEM_TEMPLATE
        else:
            template = self.CODE_PROBLEM_TEMPLATE
            
        # 格式化 prompt
        prompt = template.format(
            concepts="\n".join(f"- {c}" for c in concepts),
            difficulty=difficulty
        )
        
        # 生成
        response = self._generate_with_llm(prompt)
        
        # 解析
        data = self._parse_json_response(response)
        if data is None:
            return None
            
        return Problem(
            id=self._generate_id(),
            concepts=concepts,
            difficulty=difficulty,
            problem_text=data.get("problem", ""),
            answer=data.get("answer"),
            solution=data.get("solution"),
            metadata={
                "raw_response": response,
                "problem_type": problem_type
            }
        )
        
    def batch_generate(
        self, 
        concept_pairs: List[Dict],
        problem_type: str = "math",
        max_items: Optional[int] = None,
        verbose: bool = True
    ) -> List[Problem]:
        """
        批量生成问题
        
        Args:
            concept_pairs: 概念对列表 (每个元素包含 concepts 和 difficulty)
            problem_type: 问题类型
            max_items: 最大生成数量
            verbose: 是否显示进度
            
        Returns:
            生成的问题列表
        """
        max_items = max_items or len(concept_pairs)
        concept_pairs = concept_pairs[:max_items]
        
        problems = []
        if verbose:
            pbar = tqdm(total=len(concept_pairs), desc="生成问题")
            
        for pair in concept_pairs:
            problem = self.generate_problem(
                concepts=pair.get("concepts", []),
                difficulty=pair.get("difficulty", "medium"),
                problem_type=problem_type
            )
            
            if problem is not None:
                problems.append(problem)
                
            if verbose:
                pbar.update(1)
                pbar.set_postfix({
                    "成功": f"{len(problems)}/{len(concept_pairs)}"
                })
                
        if verbose:
            pbar.close()
            
        return problems
        
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        import time
        import random
        return f"prob_{int(time.time() * 1000)}_{random.randint(0, 9999)}"
        
    def save_problems(
        self, 
        problems: List[Problem], 
        output_path: str
    ):
        """
        保存问题到文件
        
        Args:
            problems: 问题列表
            output_path: 输出文件路径
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for problem in problems:
                json.dump(problem.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
    def load_problems(self, input_path: str) -> List[Problem]:
        """
        从文件加载问题
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            问题列表
        """
        problems = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                problems.append(Problem(
                    id=data["id"],
                    concepts=data.get("concepts", []),
                    difficulty=data.get("difficulty", "medium"),
                    problem_text=data["problem"],
                    answer=data.get("answer"),
                    solution=data.get("solution"),
                    metadata=data.get("metadata", {})
                ))
        return problems


# 延迟导入 torch
import torch


if __name__ == "__main__":
    # 测试
    generator = ProblemGenerator(use_vllm=False)
    
    # 生成单个问题
    concepts = ["数列", "极限", "微积分"]
    problem = generator.generate_problem(concepts, "medium", "math")
    
    if problem:
        print(f"问题 ID: {problem.id}")
        print(f"难度: {problem.difficulty}")
        print(f"问题: {problem.problem_text}")
        print(f"答案: {problem.answer}")
