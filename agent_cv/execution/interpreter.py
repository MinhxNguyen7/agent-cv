import traceback
import logging

import nbformat
import nbclient
from pydantic import BaseModel, Field

from agent_cv.utils import normalize_markup, create_logger


class ExecutionResult(BaseModel):
    """
    Model to hold execution logs.
    """

    stdout: str = Field(default="", description="Standard output from the execution")
    stderr: str = Field(default="", description="Standard error from the execution")
    exit_code: int = Field(default=0, description="Exit code of the execution")

    def __str__(self):
        return normalize_markup(
            f"""
        <ExecutionResult>
        code: {self.exit_code}
        <stdout>
        {self.stdout}
        </stdout>
        <stderr>
        {self.stderr}
        </stderr>
        </ExecutionResult>
        """
        )


class CodeInterpreter:
    def __init__(self):
        self._notebook, self._client = self._init_notebook()

        self._logger = create_logger(
            name="CodeInterpreter",
            level=logging.INFO,
            console_output=True,
        )

        self._logger.info("Initialized CodeInterpreter with a new Jupyter notebook.")

    @staticmethod
    def _init_notebook() -> tuple[nbformat.NotebookNode, nbclient.NotebookClient]:
        """
        Initialize a new Jupyter notebook and client.
        """
        nb = nbformat.v4.new_notebook()
        client = nbclient.NotebookClient(nb)

        client.create_kernel_manager()
        client.start_new_kernel()
        client.start_new_kernel_client()

        return nb, client

    @staticmethod
    def _make_code_cell(code: str) -> nbformat.NotebookNode:
        """
        Create a new code cell for the notebook.
        """
        return nbformat.v4.new_code_cell(code)

    @staticmethod
    def _parse_cell_outputs(cell: nbformat.NotebookNode) -> ExecutionResult:
        """
        Parse the outputs from an executed cell.
        """
        stdout_parts = []
        stderr_parts = []
        exit_code = 0

        for output in cell.outputs:
            if output.output_type == "stream":
                if output.name == "stdout":
                    stdout_parts.append(output.text)
                elif output.name == "stderr":
                    stderr_parts.append(output.text)
            elif output.output_type == "error":
                stderr_parts.extend(output.traceback)
                exit_code = 1

        return ExecutionResult(
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            exit_code=exit_code,
        )

    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a Jupyter notebook environment.
        """

        try:
            self._logger.info(f"Executing code:\n<Code>\n{code}\n</Code>")

            cell = self._make_code_cell(code)
            self._notebook.cells.append(cell)

            self._client.execute_cell(cell, len(self._notebook.cells) - 1)

            result = self._parse_cell_outputs(cell)

            self._logger.info(f"Execution result:\n{result}")

            return result

        except Exception as e:
            self._logger.error(f"Error executing code:\n{e}")
            self._logger.error(f"Full traceback:\n{traceback.format_exc()}")

            return ExecutionResult(stdout="", stderr=str(e), exit_code=1)
