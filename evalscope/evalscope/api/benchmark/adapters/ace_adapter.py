"""
ACE (Agentic Context Engineering) Adapter for evalscope.

This adapter integrates the ACE framework's self-learning mechanism into evalscope's
evaluation pipeline, enabling models to learn and improve strategies during evaluation.

Key Features:
- Generator: Uses playbook strategies to answer questions
- Reflector: Analyzes performance and identifies effective strategies
- Curator: Updates the playbook based on reflections
- Playbook: Dynamic strategy repository that evolves during evaluation
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from overrides import override

from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import Model, ModelOutput
from evalscope.api.messages import ChatMessageUser
from evalscope.utils import get_logger

logger = get_logger()

# import ACE
try:
    # 假设 agentic-context-engine 在项目路径中
    ace_path = Path(__file__).parent.parent.parent.parent.parent.parent / "agentic-context-engine"
    if ace_path.exists():
        sys.path.insert(0, str(ace_path))
    
    from ace import Generator, Reflector, Curator, Playbook, GeneratorOutput
    from ace.llm import LLMClient, LLMResponse
    ACE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ACE framework not available: {e}")
    ACE_AVAILABLE = False


class EvalscopeModelWrapper(LLMClient if ACE_AVAILABLE else object):
    """
    包装器：将 evalscope 的 Model 对象适配为 ACE 的 LLMClient 接口。
    
    ACE 框架期望 LLM 客户端有 complete(prompt) 方法，
    但 evalscope 的 Model 使用 generate() 方法。
    这个包装器桥接了两者的接口差异。
    """
    
    def __init__(self, evalscope_model: Model, generation_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            evalscope_model: evalscope 的 Model 对象
            generation_config: 生成配置（如 max_tokens, temperature 等）
        """
        if ACE_AVAILABLE:
            super().__init__(model=evalscope_model.name)
        self.evalscope_model = evalscope_model
        # 保存 generation_config，用于传递给 model.generate()
        self.generation_config = generation_config or {}
    
    def complete(self, prompt: str, **kwargs) -> 'LLMResponse':
        """
        实现 ACE 的 complete 接口，内部调用 evalscope 的 generate 方法。
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外参数（ACE 特定参数会被过滤掉）
        
        Returns:
            LLMResponse: ACE 期望的响应格式
        """
        # 过滤掉 ACE 特定的参数，这些参数 evalscope 不需要
        ace_specific_params = {'refinement_round', 'max_refinement_rounds', 'stream_thinking'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ace_specific_params}
        
        # 合并 generation_config 和 kwargs
        # kwargs 优先级更高（可以覆盖 generation_config）
        call_kwargs = {**self.generation_config, **filtered_kwargs}
        
        # 从 call_kwargs 中提取 max_tokens 等参数
        # evalscope Model.generate() 接受的参数有限，我们需要通过修改 model.config 来传递参数
        # 但为了不影响全局配置，我们创建一个临时的 config 副本
        from evalscope.api.model.generate_config import GenerateConfig
        
        # 创建临时配置（基于原配置）
        temp_config = None
        if hasattr(self.evalscope_model, 'config') and self.evalscope_model.config:
            # 复制当前配置
            config_dict = self.evalscope_model.config.model_dump()
            # 覆盖关键参数
            if 'max_tokens' in call_kwargs:
                config_dict['max_tokens'] = call_kwargs['max_tokens']
                logger.debug(f"[ACE] Setting max_tokens={call_kwargs['max_tokens']}")
            if 'temperature' in call_kwargs:
                config_dict['temperature'] = call_kwargs['temperature']
            if 'top_p' in call_kwargs:
                config_dict['top_p'] = call_kwargs['top_p']
            temp_config = GenerateConfig(**config_dict)
            logger.debug(f"[ACE] Using config: max_tokens={temp_config.max_tokens}, temp={temp_config.temperature}")
        
        # 调用 evalscope 的 generate 方法
        # 注意：ACE 的 prompt 已经要求模型返回 JSON，我们不再添加额外指令
        model_output: ModelOutput = self.evalscope_model.generate(
            input=prompt,
            config=temp_config  # 传递临时配置
        )
        
        # 提取生成的文本
        if hasattr(model_output, 'completion') and model_output.completion:
            text = str(model_output.completion)
        elif hasattr(model_output, 'message') and hasattr(model_output.message, 'content'):
            text = str(model_output.message.content)
        else:
            text = str(model_output)
        
        # 构造 ACE 期望的 LLMResponse 对象
        return LLMResponse(
            text=text,
            raw={
                'model': self.evalscope_model.name,
                'evalscope_output': model_output
            }
        )


class ACEAdapter(DefaultDataAdapter):
    """
    ACE Adapter - 将 ACE 自我学习机制集成到 evalscope 评测流程中。
    
    工作流程：
    1. 初始化：创建空 Playbook 和 ACE 三大角色
    2. 推理阶段：
       - Generator 使用 Playbook 策略生成答案
       - 记录使用的策略 ID
    3. 反思阶段：
       - Reflector 分析答案质量（对比 ground truth）
       - 标记策略为 helpful/harmful/neutral
    4. 更新阶段：
       - Curator 根据反思更新 Playbook
       - 添加新策略、更新已有策略、删除有害策略
    5. 下一轮：使用更新后的 Playbook 处理下一个样本
    
    配置参数：
        enable_adaptation (bool): 是否启用自适应学习（默认True）
        ace_model_name (str): ACE 使用的模型名称（可与评测模型不同）
        save_playbook (bool): 是否保存最终 Playbook（默认True）
        playbook_init_path (str): 初始 Playbook 路径（可选）
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if not ACE_AVAILABLE:
            raise RuntimeError(
                "ACE framework is not available. Please ensure agentic-context-engine "
                "is installed and accessible."
            )
        
        # ==================== ACE 配置 ====================
        self.enable_adaptation = kwargs.get('enable_adaptation', True)
        self.ace_model_name = kwargs.get('ace_model_name', None)  # 可用不同模型做反思
        self.save_playbook = kwargs.get('save_playbook', True)
        self.playbook_init_path = kwargs.get('playbook_init_path', None)
        
        # ==================== ACE 核心组件 ====================
        self.playbook: Optional['Playbook'] = None
        self.generator: Optional['Generator'] = None
        self.reflector: Optional['Reflector'] = None
        self.curator: Optional['Curator'] = None
        self.ace_llm_wrapper: Optional[EvalscopeModelWrapper] = None
        
        # 线程锁：保证 ACE 组件初始化的线程安全
        self._init_lock = threading.Lock()
        
        # ==================== 统计信息 ====================
        self.sample_count = 0
        self.correct_count = 0
        self.adaptation_history = []  # 记录每次 Playbook 更新
        
        logger.info("ACE Adapter initialized with adaptation=%s", self.enable_adaptation)
    
    def _initialize_ace_components(self, model: Model) -> None:
        """
        初始化 ACE 组件（延迟初始化，在第一次推理时调用）
        
        使用线程锁保证并发安全
        
        Args:
            model: evalscope 的模型对象
        """
        with self._init_lock:
            if self.playbook is not None:
                return  # 已经初始化过了
            
            logger.info("Initializing ACE components...")
            
            # 1. 初始化 Playbook
            if self.playbook_init_path and Path(self.playbook_init_path).exists():
                logger.info(f"Loading playbook from {self.playbook_init_path}")
                self.playbook = Playbook.load_from_file(self.playbook_init_path)
            else:
                logger.info("Creating empty playbook")
                self.playbook = Playbook()
            
            # 2. 创建 LLM 客户端（用于 ACE 角色）
            # 使用包装器将 evalscope 的 model 对象适配为 ACE 的 LLMClient 接口
            # 传递 generation_config，确保 max_tokens 等参数生效
            generation_config = model.config.model_dump() if hasattr(model, 'config') else {}
            self.ace_llm_wrapper = EvalscopeModelWrapper(model, generation_config)
            logger.info(f"ACE using evalscope model (wrapped): {model.name}")
            logger.info(f"Generation config: {generation_config}")
            
            # 3. 创建三大角色
            self.generator = Generator(self.ace_llm_wrapper)
            self.reflector = Reflector(self.ace_llm_wrapper)
            self.curator = Curator(self.ace_llm_wrapper)
            
            logger.info(f"ACE components initialized. Playbook has {len(self.playbook.bullets())} bullets.")
    
    @override
    def _on_inference_start(self, model: Model, sample: Sample) -> None:
        """
        推理开始前的准备工作：确保 ACE 组件已初始化
        """
        # 延迟初始化 ACE 组件
        if self.playbook is None:
            self._initialize_ace_components(model)
        
        self.sample_count += 1
        logger.debug(f"Processing sample #{self.sample_count}: {sample.id}")
    
    @override
    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """
        核心推理逻辑：使用 Generator 生成答案
        
        注意：这里不直接调用 evalscope 的 model，而是用 ACE Generator
        """
        # 确保 ACE 组件已初始化（双重检查）
        if self.generator is None:
            self._initialize_ace_components(model)
        
        # 提取问题文本
        if isinstance(sample.input, list):
            # 处理 ChatMessage 列表
            question = "\n".join([msg.content for msg in sample.input if hasattr(msg, 'content')])
        else:
            question = str(sample.input)
        
        # 使用 Generator 生成答案（带 Playbook 策略）
        generator_output: 'GeneratorOutput' = self.generator.generate(
            question=question,
            context=f"This is a {self.dataset_id} evaluation task.",
            playbook=self.playbook,
            reflection=None,  # 首次推理没有 reflection
        )
        
        # 将 GeneratorOutput 转换为 evalscope 的 ModelOutput
        # 需要正确构造 ChatCompletionChoice 和 choices 列表
        from evalscope.api.model.model_output import ChatCompletionChoice
        from evalscope.api.messages import ChatMessageAssistant
        
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(content=generator_output.final_answer),
            stop_reason='stop'
        )
        
        model_output = ModelOutput(
            model=model.name,
            choices=[choice],  # 必须设置 choices 列表
            # 保存额外信息到 metadata
            metadata={
                "ace_reasoning": generator_output.reasoning,
                "ace_bullet_ids": generator_output.bullet_ids,
            }
        )
        
        return model_output
    
    @override
    def _on_inference_end(
        self, model: Model, sample: Sample, model_output: ModelOutput, output_dir: str, **kwargs
    ) -> TaskState:
        """
        推理结束后的处理：使用 Reflector 和 Curator 更新 Playbook
        """
        # 创建 TaskState（继承父类行为）
        task_state = TaskState(
            model=model.name,
            sample=sample,
            messages=[model_output.message],
            output=model_output,
            completed=True,
        )
        
        # ==================== ACE 自适应学习部分 ====================
        if self.enable_adaptation and sample.target is not None:
            try:
                # 重新构造 GeneratorOutput（从 metadata 中恢复）
                generator_output = GeneratorOutput(
                    reasoning=model_output.metadata.get("ace_reasoning", ""),
                    final_answer=model_output.completion,
                    bullet_ids=model_output.metadata.get("ace_bullet_ids", []),
                    raw={}
                )
                
                # 1. Reflector 分析表现
                reflection = self.reflector.reflect(
                    question=str(sample.input) if isinstance(sample.input, str) 
                              else "\n".join([m.content for m in sample.input if hasattr(m, 'content')]),
                    generator_output=generator_output,
                    playbook=self.playbook,
                    ground_truth=str(sample.target),  # 正确答案
                    feedback=None,  # evalscope 没有环境反馈
                )
                
                # 2. Curator 更新 Playbook
                curator_output = self.curator.curate(
                    reflection=reflection,
                    playbook=self.playbook,
                    question_context=f"{self.dataset_id} evaluation",
                    progress=f"{self.sample_count} samples processed"
                )
                
                # 3. 应用更新
                self.playbook.apply_delta(curator_output.delta)
                
                # 4. 记录更新历史
                self.adaptation_history.append({
                    "sample_id": sample.id,
                    "delta_operations": len(curator_output.delta.operations),
                    "playbook_size": len(self.playbook.bullets()),
                })
                
                logger.debug(
                    f"Playbook updated: {len(curator_output.delta.operations)} operations, "
                    f"total bullets: {len(self.playbook.bullets())}"
                )
                
            except Exception as e:
                logger.warning(f"ACE adaptation failed for sample {sample.id}: {e}")
        
        return task_state
    
    def _on_generate_report_end(self, report, output_dir: str, **kwargs) -> None:
        """
        报告生成后：保存最终的 Playbook
        """
        super()._on_generate_report_end(report, output_dir, **kwargs)
        
        if self.save_playbook and self.playbook is not None:
            # 保存到 outputs/时间戳/ace/ 目录，避免被 evalscope 当作报告加载
            # output_dir 通常是 outputs/时间戳/reports/模型名/
            # 我们需要退回到 outputs/时间戳/ 级别
            from pathlib import Path
            output_path = Path(output_dir)
            
            # 找到 outputs 目录层级（包含时间戳的那层）
            # 从 output_dir 向上找，直到找到 reports 的父目录
            current = output_path
            while current.name != 'reports' and current.parent != current:
                current = current.parent
            
            # current 现在是 reports 目录，current.parent 是时间戳目录
            if current.name == 'reports':
                ace_output_dir = current.parent / 'ace'
            else:
                # 如果没找到 reports 目录，直接在 output_dir 的上上级创建
                ace_output_dir = output_path.parent.parent / 'ace'
            
            ace_output_dir.mkdir(parents=True, exist_ok=True)
            
            playbook_path = ace_output_dir / "final_playbook.json"
            self.playbook.save_to_file(str(playbook_path))
            logger.info(f"Final playbook saved to {playbook_path}")
            logger.info(f"Playbook stats: {self.playbook.stats()}")
            
            # 保存适应历史
            import json
            history_path = ace_output_dir / "adaptation_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.adaptation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Adaptation history saved to {history_path}")
