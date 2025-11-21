"""Workflow management for the OpenStack Upgrade Assistant."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gemini_client import GeminiClient
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Attributes:
        name: Name of the step
        type: Type of step ('ask' for AI question, 'tool' for MCP tool call)
        content: The question to ask or tool name to call
        arguments: Arguments for tool calls (only for 'tool' type)
        description: Optional description of the step
    """
    name: str
    type: str
    content: str
    arguments: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


@dataclass
class Workflow:
    """A workflow definition.

    Attributes:
        name: Name of the workflow
        description: Description of what the workflow does
        steps: List of steps to execute
        system_instruction: Optional system instruction to define agent identity and behavior
    """
    name: str
    description: str
    steps: List[WorkflowStep]
    system_instruction: Optional[str] = None

    @classmethod
    def from_file(cls, filepath: Path) -> "Workflow":
        """Load a workflow from a JSON file.

        Args:
            filepath: Path to the workflow JSON file

        Returns:
            Workflow instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Workflow file not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            steps = [
                WorkflowStep(
                    name=step["name"],
                    type=step["type"],
                    content=step["content"],
                    arguments=step.get("arguments"),
                    description=step.get("description"),
                )
                for step in data.get("steps", [])
            ]

            return cls(
                name=data["name"],
                description=data["description"],
                steps=steps,
                system_instruction=data.get("system_instruction"),
            )
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid workflow file format: {e}")


class WorkflowRunner:
    """Executes workflows using Gemini and MCP.

    Attributes:
        gemini_client: Gemini client for AI interactions
        mcp_client: MCP client for tool calls
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        mcp_client: Optional[MCPClient] = None,
    ):
        """Initialize the workflow runner.

        Args:
            gemini_client: Gemini client for AI interactions
            mcp_client: Optional MCP client for tool calls
        """
        self.gemini_client = gemini_client
        self.mcp_client = mcp_client

    async def run_workflow(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Run a workflow.

        Args:
            workflow: The workflow to execute

        Returns:
            List of results from each step

        Raises:
            RuntimeError: If a step fails
        """
        logger.info(f"Starting workflow: {workflow.name}")

        # Start chat session with system instruction if provided
        if workflow.system_instruction:
            self.gemini_client.start_chat(system_instruction=workflow.system_instruction)
            logger.info("Agent identity configured with system instruction")
        else:
            self.gemini_client.start_chat()

        results = []

        for i, step in enumerate(workflow.steps, 1):
            logger.info(f"Executing step {i}/{len(workflow.steps)}: {step.name}")

            try:
                if step.type == "ask":
                    result = await self._execute_ask_step(step)
                elif step.type == "tool":
                    result = await self._execute_tool_step(step)
                else:
                    raise ValueError(f"Unknown step type: {step.type}")

                results.append({
                    "step": step.name,
                    "type": step.type,
                    "result": result,
                })

                logger.info(f"Step {i} completed successfully")

            except Exception as e:
                logger.error(f"Step {i} failed: {e}")
                results.append({
                    "step": step.name,
                    "type": step.type,
                    "error": str(e),
                })
                raise RuntimeError(f"Workflow failed at step '{step.name}': {e}")

        logger.info(f"Workflow '{workflow.name}' completed successfully")
        return results

    async def _execute_ask_step(self, step: WorkflowStep) -> str:
        """Execute an 'ask' step.

        Args:
            step: The step to execute

        Returns:
            The AI response
        """
        logger.debug(f"Asking Gemini: {step.content}")
        response = self.gemini_client.send_message(step.content)
        return response

    async def _execute_tool_step(self, step: WorkflowStep) -> Any:
        """Execute a 'tool' step.

        Args:
            step: The step to execute

        Returns:
            The tool result

        Raises:
            RuntimeError: If MCP client is not available
        """
        if not self.mcp_client:
            raise RuntimeError(
                "MCP client is required for tool steps but was not provided"
            )

        logger.debug(f"Calling tool: {step.content} with args: {step.arguments}")
        result = await self.mcp_client.call_tool(
            step.content,
            step.arguments or {},
        )
        return result
