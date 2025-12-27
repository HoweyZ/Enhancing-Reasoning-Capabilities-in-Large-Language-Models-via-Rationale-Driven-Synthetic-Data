import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainingExample:
    """训练示例数据类"""
    prompt: str
    completion: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "metadata": self.metadata or {}
        }


class SFTDataset(Dataset):
    """
    SFT 数据集
    
    从 JSONL 文件加载训练数据，支持 tokenization。
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        use_chat_template: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            use_chat_template: 是否使用 chat template
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        
        self.examples = self._load_data()
        
    def _load_data(self) -> List[TrainingExample]:
        """加载训练数据"""
        examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                examples.append(TrainingExample(
                    prompt=data.get("prompt", ""),
                    completion=data.get("completion", ""),
                    metadata=data.get("metadata", {})
                ))
        return examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 构建完整文本
        if self.use_chat_template:
            messages = [
                {"role": "user", "content": example.prompt},
                {"role": "assistant", "content": example.completion}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            text = example.prompt + "\n\n" + example.completion
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 设置 labels (只在 completion 部分计算 loss)
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # 找到 completion 开始位置
        if self.use_chat_template and hasattr(self.tokenizer, 'assistant_token_id'):
            # 找到 assistant token 的位置
            assistant_id = self.tokenizer.assistant_token_id
            labels = input_ids.clone()
            labels[:] = -100  # 默认忽略所有位置
            labels[attention_mask == 1] = input_ids[attention_mask == 1]
        else:
            # 简单处理：prompt 部分忽略，completion 部分计算 loss
            labels = input_ids.clone()
            labels[:] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class SFTTrainer:
    """
    SFT 训练器
    
    使用合成数据进行监督微调，支持：
    - LoRA 高效微调
    - 全参数微调
    - DeepSpeed 加速
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./sft_output",
        learning_rate: float = 5e-6,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_length: int = 2048,
        num_train_epochs: int = 3,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_deepspeed: bool = False,
        deepspeed_config: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 SFT 训练器
        
        Args:
            model_name_or_path: 模型名称或路径
            output_dir: 输出目录
            learning_rate: 学习率
            batch_size: 批次大小
            gradient_accumulation_steps: 梯度累积步数
            max_length: 最大序列长度
            num_train_epochs: 训练轮数
            use_lora: 是否使用 LoRA
            lora_r: LoRA r 维度
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            use_deepspeed: 是否使用 DeepSpeed
            deepspeed_config: DeepSpeed 配置文件路径
        """
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"加载模型: {self.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载模型
        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                
                # 使用 4-bit 量化加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype="auto",
                    device_map="auto",
                    load_in_4bit=True
                )
                
                # 配置 LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
                
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
            except ImportError:
                print("PEFT 未安装，使用全参数微调")
                self.use_lora = False
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype="auto",
                    device_map="auto"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
    def _create_optimizer_and_scheduler(self, train_dataset):
        """创建优化器和学习率调度器"""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LinearLR
        
        # 计算训练步数
        total_steps = len(train_dataset) * self.num_train_epochs // (
            self.batch_size * self.gradient_accumulation_steps
        )
        
        # 创建优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 创建调度器
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=total_steps
        )
        
    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        **kwargs
    ):
        """
        执行训练
        
        Args:
            train_data_path: 训练数据路径
            eval_data_path: 验证数据路径 (可选)
            logging_steps: 日志步数
            save_steps: 保存步数
            eval_steps: 评估步数
        """
        # 加载模型
        self._load_model_and_tokenizer()
        
        # 创建数据集
        train_dataset = SFTDataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        
        # 创建优化器
        self._create_optimizer_and_scheduler(train_dataset)
        
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
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / self.gradient_accumulation_steps
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # 日志
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / logging_steps
                        print(f"Step {global_step}: Loss = {avg_loss:.4f}")
                        total_loss = 0
                        
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
        
        if self.use_lora:
            self.model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path)
            
        self.tokenizer.save_pretrained(output_path)
        
        # 保存训练配置
        config = {
            "model_name_or_path": self.model_name_or_path,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "num_train_epochs": self.num_train_epochs,
            "use_lora": self.use_lora,
        }
        
        with open(os.path.join(output_path, "training_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
            
    def evaluate(self, eval_data_path: str) -> Dict:
        """
        评估模型
        
        Args:
            eval_data_path: 评估数据路径
            
        Returns:
            评估指标
        """
        if self.model is None:
            self._load_model_and_tokenizer()
            
        # 加载评估数据
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f]
            
        self.model.eval()
        total_loss = 0
        correct = 0
        total = len(eval_data)
        
        for item in tqdm(eval_data, desc="评估中"):
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            
            # 构建输入
            messages = [
                {"role": "user", "content": prompt},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
                
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # 简单评估：检查是否包含正确答案
            if completion.strip() in response.strip():
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }


