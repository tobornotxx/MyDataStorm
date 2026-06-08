from dotenv import load_dotenv
import os
import sys
import subprocess
from typing import List, Dict, Any, Tuple, Optional
import tempfile


from utils import logger
from utils.prompts import SIMPLE_AGENT_SYSTEM_PROMPT, SIMPLE_AGENT_DEBUG_TEMPLATE, get_simple_agent_var_instruction
from utils.temp_file import get_var_storage_info, save_variable_to_temp
from utils.helper import extract_code_from_response, build_variable_preamble
from llm import OpenAILikeLLM, LLMConfig
load_dotenv()


class CodeAgent:
    """
    Code Agent - 通过 LLM 生成 Python 代码并执行来完成任务。
    
    工作原理：
    1. 将任务描述发送给 LLM，要求它生成 Python 代码（用 <code></code> 包裹）
    2. 提取代码块，保存为临时 .py 文件并用当前 Python 环境执行
    3. 如果执行成功，返回 stdout 输出作为结果
    4. 如果执行失败，将错误信息反馈给 LLM 进行 debug，重新生成代码
    5. 重复直到成功或达到 max_steps 次数限制
    
    参数:
        model: 模型名称
        api_base: API 基础 URL
        api_key: API 密钥
        additional_authorized_imports: 允许使用的额外 Python 库（仅做提示，不做强制限制）
        **kwargs: 传递给 LLMConfig 的额外参数 (temperature, top_p, seed, max_tokens 等)
    """

    def __init__(
        self,
        model: str = "",
        api_base: str = os.getenv("API_BASE_DEFAULT", ""),
        api_key: str = os.getenv("API_KEY_DEFAULT", ""),
        additional_authorized_imports: List[str] = [],
        **kwargs,
    ):
        config = LLMConfig(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 1.0),
            max_tokens=kwargs.get('max_tokens', None),
            seed=kwargs.get('seed', 42),
        )
        self.llm = OpenAILikeLLM(config=config)
        self.llm.set_system_prompt(SIMPLE_AGENT_SYSTEM_PROMPT)
        self.imports = additional_authorized_imports
        self.execution_timeout = kwargs.get('execution_timeout', 60)

    def run(
        self,
        input: str,
        max_steps: int = 3,
        additional_args: Dict[str, Any] = {},
    ) -> Optional[str]:
        """
        执行任务。
        
        参数:
            input: 用户的任务描述
            max_steps: 最大执行/重试步数（含首次执行）
            additional_args: 传递给代码的变量字典
            
        返回:
            执行成功时返回 stdout 输出字符串；失败返回 None
        """
        # 准备变量临时文件
        file_paths = {}
        var_paths = {}
        var_type_info = {}

        try:
            for key, value in additional_args.items():
                suffix, type_name = get_var_storage_info(value)
                temp_path = save_variable_to_temp(key, value, suffix, type_name)
                logger.info(f"[CodeAgent] 变量 '{key}' 保存到临时文件: {temp_path}")
                file_paths[key] = temp_path
                var_paths[key] = temp_path
                var_type_info[key] = type_name

            # 构建变量读取指令（复用已有的工具函数）
            var_instruction = get_simple_agent_var_instruction(var_type_info)
            full_query = var_instruction + input

            # 清空对话历史，开始新会话
            self.llm.clear_history()

            result = self._run_loop(full_query, var_paths, max_steps)
            return result

        except Exception as e:
            logger.error(f"[CodeAgent] 运行出错: {e}")
            return None
        finally:
            # 清理临时文件
            for temp_path in file_paths.values():
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"[CodeAgent] 无法删除临时文件 {temp_path}: {e}")

    def _run_loop(
        self,
        query: str,
        var_paths: Dict[str, str],
        max_steps: int,
    ) -> Optional[str]:
        """核心执行循环：生成代码 → 执行 → 成功则返回 / 失败则 debug 重试。"""
        
        separator = "=" * 60

        # 第 1 步：让 LLM 根据 query 生成代码
        logger.info(f"\n{separator}")
        logger.info(f"[CodeAgent] Step 1/{max_steps}: 请求 LLM 生成代码")
        logger.info(separator)
        logger.log_to_file(query, label="PROMPT")

        response = self.llm.chat(query, keep_history=True)
        code = extract_code_from_response(response.content)

        if code is None:
            logger.error(f"\n{separator}")
            logger.error("[CodeAgent] LLM 未返回有效的 <code></code> 代码块")
            logger.error(separator)
            logger.log_to_file(response.content, label="LLM_RAW_RESPONSE")
            return None

        for step in range(1, max_steps + 1):
            logger.info(f"\n{separator}")
            logger.info(f"[CodeAgent] Step {step}/{max_steps}: 执行代码")
            logger.info(separator)
            logger.log_to_file(code, label="CODE")

            success, output = self._execute_code(code, var_paths)

            if success:
                logger.info(f"\n{separator}")
                logger.info(f"[CodeAgent] ✓ 代码执行成功 (Step {step}/{max_steps})")
                logger.info(separator)
                logger.log_to_file(output.strip(), label="RESULT")
                return output.strip() if output else ""

            # 执行失败，记录错误
            logger.warning(f"\n{separator}")
            logger.warning(f"[CodeAgent] ✗ Step {step}/{max_steps} 代码执行失败")
            logger.warning(separator)
            logger.log_to_file(output, label="ERROR")

            # 如果已达到最大步数，不再重试
            if step >= max_steps:
                logger.error(f"\n{separator}")
                logger.error(f"[CodeAgent] 已达到最大步数 {max_steps}，停止重试")
                logger.error(separator)
                return None

            # 让 LLM debug 并生成新代码
            logger.info(f"\n{separator}")
            logger.info(f"[CodeAgent] Step {step}→{step+1}: 请求 LLM 修复代码")
            logger.info(separator)
            debug_msg = SIMPLE_AGENT_DEBUG_TEMPLATE.format(code=code, error=output)
            logger.log_to_file(debug_msg, label="DEBUG_PROMPT")
            
            response = self.llm.chat(debug_msg, keep_history=True)
            new_code = extract_code_from_response(response.content)

            if new_code is None:
                logger.error("[CodeAgent] LLM debug 后未返回有效代码块")
                logger.log_to_file(response.content, label="LLM_DEBUG_RAW")
                return None

            code = new_code

        return None

    # final_answer 函数注入代码：让 LLM 生成的 final_answer() 调用等价于 print()
    _FINAL_ANSWER_SHIM = (
        "def final_answer(result):\n"
        "    \"\"\"内置函数：返回最终结果。\"\"\"\n"
        "    print(result)\n"
    )

    def _execute_code(
        self, code: str, var_paths: Dict[str, str]
    ) -> Tuple[bool, str]:
        """将代码保存到临时 .py 文件并用当前 Python 环境执行。
        
        返回:
            (success: bool, output: str) - 成功时 output 为 stdout，失败时为 stderr
        """
        # 在代码顶部注入 final_answer shim + 变量路径赋值
        preamble_parts = [self._FINAL_ANSWER_SHIM]
        var_preamble = build_variable_preamble(var_paths)
        if var_preamble:
            preamble_parts.append(var_preamble)
        preamble = "\n".join(preamble_parts)
        full_code = preamble + "\n\n" + code

        # 写入临时 .py 文件
        temp_fd, temp_script = tempfile.mkstemp(suffix='.py', prefix='simple_agent_')
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(full_code)

            # 使用当前 Python 解释器执行
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=os.getcwd(),
            )

            if result.returncode == 0:
                return True, result.stdout
            else:
                # 合并 stderr 和 stdout（有些错误信息可能在 stdout 里）
                error_output = result.stderr
                if result.stdout:
                    error_output = f"stdout:\n{result.stdout}\nstderr:\n{error_output}"
                return False, error_output

        except subprocess.TimeoutExpired:
            return False, f"代码执行超时（超过 {self.execution_timeout} 秒）"
        except Exception as e:
            return False, f"执行代码时出现异常: {type(e).__name__}: {e}"
        finally:
            try:
                os.remove(temp_script)
            except OSError:
                pass


def create_code_agent(
    model: str,
    api_base: str = os.getenv("API_BASE_DEFAULT"),
    api_key: str = os.getenv("API_KEY_DEFAULT"),
    additional_authorized_imports: List[str] = [],
    **kwargs,
) -> CodeAgent:
    """
    工厂函数：创建 CodeAgent 实例。
    
    参数:
        model: 模型名称
        api_base: API 基础 URL
        api_key: API 密钥
        additional_authorized_imports: 允许使用的额外 Python 库
        **kwargs: 传递给 CodeAgent 的额外参数
    
    返回:
        CodeAgent 实例
    """
    # 忽略旧接口可能传入的不支持参数
    kwargs.pop("max_print_outputs_length", None)
    kwargs.pop("tools", None)
    return CodeAgent(
        model=model,
        api_base=api_base,
        api_key=api_key,
        additional_authorized_imports=additional_authorized_imports,
        **kwargs,
    )


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("测试 CodeAgent")
    print("=" * 60)
    agent = CodeAgent(
        model=os.getenv("CODE_AGENT_MODEL_NAME", "siliconflow/Qwen/Qwen3-8B"),
        api_base=os.getenv("API_BASE_DEFAULT"),
        api_key=os.getenv("API_KEY_DEFAULT"),
        additional_authorized_imports=['pandas', 'numpy'],
    )
    result = agent.run(
        "按照以下方式计算，并返回最终结果：1. list中所有数字的累加求和; 2. 减去pandas dataframe的平均值; 3. 再减去numpy array的平均值",
        max_steps=3,
        additional_args={'integer_list':[i for i in range(1, 101)], 'pandas_dataframe': pd.DataFrame({'a': [1, 2, 3, 4, 5]}), 'numpy_array': np.array([1, 2, 3, 4, 5])}
    )
    print(f"CodeAgent result: {result}")  # 应该是5050-3-3=5044