import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SelfPlayExample:
    """Self-Play 训练示例"""
    problem_id: str
    problem: str
    chosen_response: str  # 正确的解答
    rejected_response: str  # 错误的解答
    chosen_correct: bool
    rejected_correct: bool
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "problem_id": self.problem_id,
            "problem": self.problem,
            "chosen": self.chosen_response,
            "rejected": self.rejected_response,
            "chosen_correct": self.chosen_correct,
            "rejected_correct": self.rejected_correct,
            "metadata": self.metadata or {}
        }


class SelfPlayDataset(Dataset):
    """
    Self-Play 数据集
    
    加载 chosen/rejected 对用于 DPO 训练。
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.examples = self._load_data()
        
    def _load_data(self) -> List[SelfPlayExample]:
        """加载训练数据"""
        examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                examples.append(SelfPlayExample(
                    problem_id=data.get("problem_id", ""),
                    problem=data.get("problem", ""),
                    chosen_response=data.get("chosen", ""),
                    rejected_response=data.get("rejected", ""),
                    chosen_correct=data.get("chosen_correct", True),
                    rejected_correct=data.get("rejected_correct", False),
                    metadata=data.get("metadata", {})
                ))
        return examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 构建 chosen 和 rejected 的输入
        chosen_text = example.problem + "\n\n" + example.chosen_response
        rejected_text = example.problem + "\n\n" + example.rejected_response
        
        # Tokenize
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
        }


class SelfPlayTrainer:
    """
    Self-Play 训练器
    
    基于自生成数据进行训练，支持：
    - DPO (Direct Preference Optimization)
    - 简单的 preference 训练
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./selfplay_output",
        learning_rate: float = 1e-6,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_length: int = 2048,
        num_train_epochs: int = 3,
        beta: float = 0.1,  # DPO temperature
        **kwargs
    ):
        """
        初始化 Self-Play 训练器
        
        Args:
            model_name_or_path: 模型名称或路径
            output_dir: 输出目录
            learning_rate: 学习率
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_length: 最大序列长度
            num_train_epochs: 训练轮数
            beta: DPO temperature 参数
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.beta = beta
        
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.optimizer = None
        
    def _load_model_and_tokenizer(self):
        """加载模型和参考模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"加载模型: {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 加载参考模型 (用于 DPO)
        print("加载参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.ref_model.eval()
        
        # 冻结参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def _compute_dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 DPO loss
        
        Args:
            chosen_logps: chosen 样本的 log probabilities
            rejected_logps: rejected 样本的 log probabilities
            ref_chosen_logps: 参考模型的 chosen log probabilities
            ref_rejected_logps: 参考模型的 rejected log probabilities
            
        Returns:
            DPO loss
        """
        # 计算 preference 损失
        # log(sigmoid(beta * (logps_chosen - logps_rejected)))
        # 简化为 hinge loss 形式
        
        # 简化的 DPO loss
        differences = (chosen_logps - rejected_logps) - (
            ref_chosen_logps - ref_rejected_logps
        )
        
        loss = -torch.nn.functional.logsigmoid(self.beta * differences).mean()
        
        return loss
        
    def _compute_logps(
        self, 
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列的 log probabilities
        
        Args:
            model: 语言模型
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            
        Returns:
            每个序列的 log probability
        """
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        # 计算 token 级别的 log prob
        log_probs = output.logits.log_softmax(dim=-1)
        
        # 移位：预测下一个 token
        log_probs = log_probs[..., :-1, :].contiguous()
        input_ids = input_ids[..., 1:].contiguous()
        attention_mask = attention_mask[..., 1:].contiguous()
        
        # 选择对应 token 的 log prob
        log_probs = torch.gather(
            log_probs, 
            2, 
            input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # 应用掩码
        log_probs = log_probs * attention_mask
        
        # 序列级别的平均 log prob
        logps = log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)
        
        return logps
        
    def prepare_training_data(
        self,
        problems: List[Dict],
        output_path: str = "data/selfplay_pairs.jsonl",
        **kwargs
    ) -> List[SelfPlayExample]:
        """
        准备 Self-Play 训练数据
        
        Args:
            problems: 问题列表
            output_path: 输出文件路径
            
        Returns:
            Self-Play 训练示例列表
        """
        from ..data_generation.trajectory_collector import TrajectoryCollector
        from ..data_generation.evaluator import Evaluator
        
        collector = TrajectoryCollector(
            model_path=self.model_name_or_path,
            temperature=0.7
        )
        evaluator = Evaluator(eval_type="math")
        
        training_examples = []
        
        for problem in tqdm(problems, desc="准备训练数据"):
            # 收集多个解答
            trajectories = collector.collect_single(
                problem["id"],
                problem["problem"],
                "math"
            )
            
            # 评估解答正确性
            if problem.get("answer"):
                eval_results = evaluator.batch_evaluate([
                    {"id": problem["id"], "prediction": t.answer, "ground_truth": problem["answer"]}
                    for t in [trajectories]
                ])
                
                # 构建 chosen/rejected 对
                if eval_results[0].is_correct:
                    example = SelfPlayExample(
                        problem_id=problem["id"],
                        problem=problem["problem"],
                        chosen_response=trajectories.response,
                        rejected_response="",  # 没有错误解答
                        chosen_correct=True,
                        rejected_correct=False,
                        metadata={
                            "chosen_answer": trajectories.answer
                        }
                    )
                    training_examples.append(example)
                    
        # 保存训练数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_examples:
                json.dump(example.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        print(f"生成了 {len(training_examples)} 个训练示例")
        return training_examples
        
    def train(
        self,
        train_data_path: str,
        logging_steps: int = 10,
        save_steps: int = 500,
        **kwargs
    ):
        """
        执行训练
        
        Args:
            train_data_path: 训练数据路径
            logging_steps: 日志步数
            save_steps: 保存步数
        """
        # 加载模型
        self._load_model_and_tokenizer()
        
        # 创建数据集
        train_dataset = SelfPlayDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        
        # 创建优化器
        from torch.optim import AdamW
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练循环
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(self.num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_train_epochs}")
            pbar = tqdm(train_loader, desc="训练中")
            
            for step, batch in enumerate(pbar):
                # 获取数据
                chosen_ids = batch["chosen_input_ids"].to(self.model.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.model.device)
                rejected_ids = batch["rejected_input_ids"].to(self.model.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.model.device)
                
                # 计算模型 logps
                chosen_logps = self._compute_logps(self.model, chosen_ids, chosen_mask)
                rejected_logps = self._compute_logps(self.model, rejected_ids, rejected_mask)
                
                # 计算参考模型 logps (冻结)
                with torch.no_grad():
                    ref_chosen_logps = self._compute_logps(
                        self.ref_model, chosen_ids, chosen_mask
                    )
                    ref_rejected_logps = self._compute_logps(
                        self.ref_model, rejected_ids, rejected_mask
                    )
                    
                # 计算 loss
                loss = self._compute_dpo_loss(
                    chosen_logps, rejected_logps,
                    ref_chosen_logps, ref_rejected_logps
                )
                loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 日志
                    if global_step % logging_steps == 0:
                        print(f"Step {global_step}: Loss = {loss.item():.4f}")
                        
                    # 保存
                    if global_step % save_steps == 0:
                        self.save_model(os.path.join(
                            self.output_dir, 
                            f"checkpoint-{global_step}"
                        ))
                        
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": global_step
                })
                
        # 保存最终模型
        self.save_model(self.output_dir)
        print(f"\n训练完成！模型保存到: {self.output_dir}")
        
    def save_model(self, output_path: str):
        """
        保存模型
        
        Args:
            output_path: 输出路径
        """
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # 保存训练配置
        config = {
            "model_name_or_path": self.model_name_or_path,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "num_train_epochs": self.num_train_epochs,
            "beta": self.beta,
        }
        
        with open(os.path.join(output_path, "training_config.json"), 'w') as f:
            json.dump(config, f, indent=2)


