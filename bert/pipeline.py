from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from plaintext_writer import PlaintextWriter


pipeline_exec = LocalPipelineExecutor(
    pipeline=[
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/10BT"),
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"),
        PlaintextWriter(output_folder="./extracted-plaintext-corpus", perform_sentence_splitting=True, sentence_splitting_language="english", compression=None),
    ],
    tasks=15,
    workers=15,
)
pipeline_exec.run()
