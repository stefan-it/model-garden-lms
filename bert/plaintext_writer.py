from typing import IO, Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter

from nltk.tokenize import sent_tokenize


class PlaintextWriter(DiskWriter):
    """Write plain text data to datafolder (local or remote)

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        perform_sentence_splitting: whether to perform sentence splitting or not. Enabled by default
        sentence_splitting_language: defines the language used for sentence splitting. Set to english by default
        compression: if any compression scheme should be used. By default, no compression is used
        adapter: a custom function to "adapt" the Document format to the desired output format
    """

    default_output_filename: str = "${rank}.txt"
    name = "📑 Plaintext writer"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        perform_sentence_splitting: bool = True,
        sentence_splitting_language: str = "english",
        compression: str | None = None,
        adapter: Callable = None,
    ):
        self.perform_sentence_splitting = perform_sentence_splitting
        self.sentence_splitting_language = sentence_splitting_language
        super().__init__(
            output_folder,
            output_filename=output_filename,
            compression=compression,
            adapter=adapter,
            mode="wt",
            max_file_size=-1,
        )

    def _write(self, document: dict, file_handler: IO, _filename: str):
        import re

        content = document["text"]

        if self.perform_sentence_splitting:
            content = re.sub(r"\n+", " ", content).strip()
            sentences = sent_tokenize(content, self.sentence_splitting_language)
            file_handler.write("\n".join(sentences) + "\n\n")
        else:
            content = re.sub(r"\n+", "\n", content).strip()
            file_handler.write(content + "\n\n")
