from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    WebToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType
from camel.logger import set_log_level

from utils import OwlRolePlaying, run_society

load_dotenv()
set_log_level(level="DEBUG")

def construct_society(question: str) -> OwlRolePlaying:
    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.GOOGLE,
            model_type="gemini-pro",
            model_config_dict={"temperature": 0.4},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.GOOGLE,
            model_type="gemini-pro",
            model_config_dict={"temperature": 0.4},
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.GOOGLE,
            model_type="gemini-pro",
            model_config_dict={"temperature": 0.4},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.GOOGLE,
            model_type="gemini-pro",
            model_config_dict={"temperature": 0.4},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.GOOGLE,
            model_type="gemini-pro-vision",
            model_config_dict={"temperature": 0.4},
        ),
    }

    # Configure toolkits
    tools = [
        *WebToolkit(
            headless=False,
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society

def main():
    # Example research question
    question = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."

    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    print(f"\nFinal Answer: {answer}")
    print(f"Total Tokens Used: {token_count}")

if __name__ == "__main__":
    main()